import sys
import os
from loguru import logger



class MyLogger:
    def __init__(self):
        self.logger = logger
        self.logger.remove()  # 清空所有默认处理器
        self.registered_tags = set()  # 记录已注册的标签

        # 创建 logs 文件夹
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # # # 控制台输出
        # self.logger.add(sys.stdout,
        #                 format="<green>{time:YYYYMMDD HH:mm:ss}</green> | "
        #                        "<cyan>{module}</cyan>.<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        #                        "<level>{level}</level>: <level>{message}</level>")

        # 保存 INFO 日志
        self.logger.add(os.path.join(log_dir, "{time:YYYYMMDD}_info.log"),
                        level="INFO",
                        filter=lambda record: record["level"].name == "INFO",
                        format="{time:YYYYMMDD HH:mm:ss} - {module}.{function}:{line}_{level}_{message}",
                        rotation="00:00")

        # 保存 ERROR 日志
        self.logger.add(os.path.join(log_dir, "{time:YYYYMMDD}_error.log"),
                        level="ERROR",
                        filter=lambda record: record["level"].name == "ERROR",
                        format="{time:YYYYMMDD HH:mm:ss} - {module}.{function}:{line}_{level}_{message}",
                        rotation="00:00")

    def register_tag(self, tag_name):
        """注册自定义标签，为指定标签创建独立的日志文件"""
        if tag_name in self.registered_tags:
            return  # 已经注册过，直接返回

        log_dir = "logs"

        # 为自定义标签创建日志文件
        self.logger.add(
            os.path.join(log_dir, "{time:YYYYMMDD}_" + tag_name + ".log"),
            level="INFO",
            filter=lambda record: "tag" in record["extra"] and record["extra"]["tag"] == tag_name,
            format="{time:YYYYMMDD HH:mm:ss} - {module}.{function}:{line}_{level}_{message}",
            rotation="00:00"
        )

        self.registered_tags.add(tag_name)
        print(f"已注册标签: {tag_name}")

    def get_logger(self):
        return self.logger

    def get_tag_logger(self, tag_name):
        """获取带有指定标签的logger实例"""
        self.register_tag(tag_name)  # 确保标签已注册
        return self.logger.bind(tag=tag_name)


# 初始化全局日志器
mylog = MyLogger().get_logger()


# 获取带标签的日志器示例
def get_sftp_logger():
    """获取SFTP专用的日志器"""
    return MyLogger().get_tag_logger("sftp")


# # 使用示例
# if __name__ == "__main__":
#     # 普通日志
#     mylog.info("这是一条普通日志")
#
#     # SFTP专用日志
#     sftp_logger = get_sftp_logger()
#     sftp_logger.info("这是一条SFTP操作日志")
#     sftp_logger.error("SFTP连接错误")
#
#     # 可以创建其他标签的日志器
#     db_logger = MyLogger().get_tag_logger("database")
#     db_logger.info("数据库操作日志")