from openvino.inference_engine import IECore

ie=IECore().available_devices
print(ie)
