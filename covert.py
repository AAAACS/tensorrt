import torch
import tensorrt as trt
from vgg16 import VGGnet

def convert_to_tensorrt():
    # 加载 PyTorch 模型
    model = VGGnet(num_classes=6)
    model.load_state_dict(torch.load('best-1229.pt'))
    model.eval().cuda()
    
    # 创建 logger 和 builder
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    
    # 创建网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # 设置最大工作空间
    config.max_workspace_size = 1 << 28  # 1GB
    
    # 允许 FP16 精度
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # 导出为 ONNX 作为中间格式
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape).cuda()
    torch.onnx.export(
        model,
        dummy_input,
        "temp.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )
    
    # 解析 ONNX 文件
    parser = trt.OnnxParser(network, logger)
    with open("temp.onnx", 'rb') as f:
        parser.parse(f.read())
    
    # 构建引擎
    print("Building TensorRT engine... This might take a few minutes.")
    engine = builder.build_engine(network, config)
    
    # 保存引擎
    with open("model.engine", "wb") as f:
        f.write(engine.serialize())
    
    print("TensorRT engine has been created and saved as 'model.engine'")

if __name__ == "__main__":
    convert_to_tensorrt() 
