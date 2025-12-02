import time
from onnxocr.onnx_paddleocr import ONNXPaddleOcr
from operator import itemgetter
import pandas as pd
import os
import json
import cv2
from onnx_detector import ONNX_Detector


def load_config():
    """加载配置文件"""
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    print("成功加载配置文件 config.json")
    return config


# 加载全局配置
config = load_config()
# 初始化模型（延迟到实际使用时）
ocr_model = None
table_model = None


def initialize_model():
    """延迟初始化模型"""
    global ocr_model
    global table_model
    if ocr_model is None and table_model is None:
        model_settings = config["model_settings"]

        print("正在初始化 OCR 模型...")
        ocr_model = ONNXPaddleOcr()
        print("OCR 模型�成功")
        table_model = ONNX_Detector(model_settings["det_model_path"], 640,
                                    model_settings["det_model_confident_threshold"])
        print("表格检测模型成功")
        return True


def get_cells(img):
    """检测图像中的单元格"""
    cells = table_model.predict(img)

    print(f"原始检测到 {len(cells)} 个单元格contour")
    print(f"Cell contour索引范围: 0-{len(cells) - 1}")
    return cells


# -------------------------
# 4. 使用层次结构信息组织6x8网格
# -------------------------


def organize_cells_by_x_columns(cells):
    """根据 x 坐标聚类 -> 分列，再在每列内部按 y 排序。
    cells 为 (x1, y1, x2, y2)
    """
    grid_config = config["grid_detection"]
    col_thresh = grid_config["row_threshold"]  # ← 改为列的阈值，仍使用 row_threshold

    cells_with_center = []
    for idx, (x1, y1, x2, y2) in enumerate(cells):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        cells_with_center.append((idx, cx, cy, x1, y1, x2, y2))

    # -------- ① 按 cx 排序（用于从左到右分列）--------
    cells_sorted = sorted(cells_with_center, key=itemgetter(1))

    columns = []
    current_col = []
    current_x = None

    # -------- ② 聚类成“列” --------
    for cell in cells_sorted:
        idx, cx, cy, x1, y1, x2, y2 = cell

        if current_x is None:
            current_x = cx
            current_col.append(cell)

        elif abs(cx - current_x) <= col_thresh:
            current_col.append(cell)

        else:
            columns.append(current_col)
            current_col = [cell]
            current_x = cx

    if current_col:
        columns.append(current_col)

    # -------- ③ 每一列内部按 y 排序（从上到下）--------
    for col in columns:
        col.sort(key=itemgetter(2))  # cy 升序

    print(f"共分成 {len(columns)} 列")
    for i, col in enumerate(columns):
        print(f"第{i}列有 {len(col)} 个单元格")

    # -------- ④ 输出原来的 box 格式 --------
    columns_cells = []
    for col in columns:
        col_cells = [(x1, y1, x2, y2) for _, cx, cy, x1, y1, x2, y2 in col]
        columns_cells.append(col_cells)

    return columns_cells


# -------------------------
# 5. 对网格进行OCR识别
# -------------------------
def ocr_grid(grid_cells, img):
    """对网格进行OCR识别（过滤低置信度和非数字结果）"""
    file_config = config["file_settings"]

    print("\n开始OCR识别...")
    results = []

    for col_idx, col in enumerate(grid_cells):
        col_results = []
        for row_idx, box in enumerate(col):
            if box is None:
                col_results.append({"text": "", "confidence": 0})
                continue

            try:
                x1, y1, x2, y2 = box
                cell_img = img[y1:y2, x1:x2]

                result = ocr_model.ocr([cell_img], cls=False)

                # PaddleOCR 结果格式：[(text, score)]
                text = result[0][0]
                confidence = result[0][1]

                # ---------- ★ 过滤 低置信度 ★ ----------
                if confidence < file_config["ocr_save_threshold"]:
                    clean_text = ""
                else:
                    # ---------- ★ 过滤 非数字内容 ★ ----------
                    try:
                        int(text)
                        clean_text = text  # 是数字
                    except:
                        clean_text = ""  # 非数字 → 丢弃

                col_results.append({"text": clean_text, "confidence": confidence})

                # 保存调试图片
                if file_config["save_debug_images"]:
                    cv2.imwrite(
                        f"images/grid_cell_{int(time.time())}_{row_idx}_{col_idx}.png",
                        cell_img
                    )

                print(f"位置(列{col_idx}, 行{row_idx}): '{text}' -> '{clean_text}' 置信度: {confidence:.2f}")


            except Exception as e:
                print(f"位置(列{col_idx}, 行{row_idx}) 识别失败: {e}")
                col_results.append({"text": "", "confidence": 0})

        results.append(col_results)

    return results


def save_csv(results, csv_filename):
    """保存结果到CSV文件"""
    file_config = config["file_settings"]
    suffix = file_config["output_csv_suffix"]

    # 也保存原始识别结果
    csv_original_filename = f"{csv_filename}{suffix}.csv"
    # -------★ 转置：列列表 → 行列表 ★--------
    # ---- 补齐不同列的长度 ----
    max_len = max(len(col) for col in results)
    padded = [col + [{}] * (max_len - len(col)) for col in results]

    # 真正的转置
    transposed = list(zip(*padded))

    # ---- 提取文本 ----
    csv_data = []
    for row in transposed:
        csv_data.append([cell.get("text", "") for cell in row])

    # ---- 列名与行名 ----
    df_original = pd.DataFrame(
        csv_data,
        columns=[f"Col_{i+1}" for i in range(len(results))],
        index=[f"Row_{i+1}" for i in range(len(csv_data))]
    )

    df_original.to_csv(csv_original_filename, index=True)
    print(f"原始识别结果CSV文件已保存为: {csv_original_filename}")


def create_images_directory():
    """创建images目录"""
    file_config = config["file_settings"]
    if file_config["save_debug_images"]:
        if not os.path.exists("images"):
            os.makedirs("images")
            print("创建调试图像目录: images")


# -------------------------
# 1. 加载图像
# -------------------------
def main():
    """主函数"""
    global config

    initialize_model()
    # 创建必要的目录
    create_images_directory()

    # 查找当前目录下所有支持的图片
    file_config = config["file_settings"]
    supported_extensions = file_config["supported_extensions"]

    image_files = []
    for filename in os.listdir("."):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            image_files.append(filename)

    if not image_files:
        print(f"错误：当前目录没有找到支持的图像文件: {supported_extensions}")
        print("请将支持的图像文件放在程序同目录下")
        input("按任意键退出...")
        return

    print(f"\n找到 {len(image_files)} 个图像文件: {image_files}")

    for image_name in image_files:
        print(f"\n正在处理图像: {image_name}")
        img = cv2.imread(image_name)

        if img is None:
            print(f"错误：无法读取图像文件 {image_name}")
            continue

        cells = get_cells(img)

        grid_cells = organize_cells_by_x_columns(cells)

        if not grid_cells or len(grid_cells) == 0:
            print("警告：未检测到有效的单元格网格")
            continue

        ocr_results = ocr_grid(grid_cells, img)

        # 保存结果，去掉文件扩展名
        base_name = os.path.splitext(image_name)[0]
        save_csv(ocr_results, base_name)

    print("\n=== 所有图像处理完成！ ===")
    print(f"配置文件: config.json")
    print(f"调试图像目录: images/")
    print(f"结果文件格式: [图像名].csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序发生未预期的错误: {e}")
        print("请检查配置文件或图像文件")
    finally:
        if 'model' in globals() and ocr_model is not None:
            print("\n正在清理资源...")
            # 可以在这里添加模型清理代码
        input("按任意键退出...")
