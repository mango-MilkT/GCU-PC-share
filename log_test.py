import logging
import time

filename = f"./logger/{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}.txt"
logger = logging.getLogger('VGG')
logger.setLevel(logging.INFO)

# 创建文件处理器，并设置级别为INFO
file_handler = logging.FileHandler(filename)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 添加文件处理器到logger
logger.addHandler(file_handler)

# 记录一些信息
logger.info('这是第一条日志信息')
logger.warning('这是一条警告信息')

# 移除文件处理器
logger.removeHandler(file_handler)