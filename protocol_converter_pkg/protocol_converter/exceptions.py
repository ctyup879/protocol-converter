"""
协议转换器自定义异常

提供细粒度的错误类型，便于调用方区分不同失败场景并采取相应策略。
"""


class ProtocolConversionError(Exception):
    """协议转换基础异常"""
    
    def __init__(self, message: str, source_protocol: str = "", target_protocol: str = ""):
        self.source_protocol = source_protocol
        self.target_protocol = target_protocol
        super().__init__(message)


class UnsupportedProtocolError(ProtocolConversionError):
    """不支持的协议类型"""
    pass


class InvalidRequestError(ProtocolConversionError):
    """无效的请求数据"""
    
    def __init__(self, message: str, field: str = "", source_protocol: str = "", target_protocol: str = ""):
        self.field = field
        super().__init__(message, source_protocol, target_protocol)


class StreamStateError(ProtocolConversionError):
    """流式转换状态异常（如并发冲突、状态未重置等）"""
    pass
