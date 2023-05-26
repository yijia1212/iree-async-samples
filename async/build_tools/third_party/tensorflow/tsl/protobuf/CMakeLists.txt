set(TENSORFLOW_ROOT "${OPENXLA_ASYNC_ROOT_DIR}/third_party/tensorflow/")

file(GLOB ProtoFiles "${TENSORFLOW_ROOT}/tensorflow/tsl/protobuf/*.proto")
PROTOBUF_GENERATE_CPP(ProtoSources ProtoHeaders ${ProtoFiles})

