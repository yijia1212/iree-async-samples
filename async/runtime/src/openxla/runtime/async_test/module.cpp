// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/async_test/module.h"

#include "iree/base/status.h"
#include "iree/vm/dynamic/api.h"
#include "iree/vm/native_module_cc.h"
#include "iree/vm/ref_cc.h"
#include "openxla/runtime/async/api.h"
#include "openxla/runtime/async/async_runtime_cc.h"
#include "tensorflow/tsl/concurrency/async_value_ref.h"

namespace openxla::runtime::asynctest {

using namespace iree;

//===----------------------------------------------------------------------===//
// AsyncTestModule state encapsulates all the state required for running
// AsyncTest operations at run time
//===----------------------------------------------------------------------===//

class AsyncTestModuleState {
 public:
  AsyncTestModuleState();
  ~AsyncTestModuleState();

  StatusOr<vm::ref<iree_async_value_t>> ReturnAsyncScalar();
};

AsyncTestModuleState::AsyncTestModuleState() {}

AsyncTestModuleState::~AsyncTestModuleState() {}

StatusOr<vm::ref<iree_async_value_t>>
AsyncTestModuleState::ReturnAsyncScalar() {
  tsl::AsyncValueRef<int32_t> value =
      tsl::MakeAvailableAsyncValueRef<int32_t>(42);
  return openxla::runtime::async::AsValue<int32_t>(value);
}

//===----------------------------------------------------------------------===//
// Functions dispatch table for AsyncTestModuleState
//===----------------------------------------------------------------------===//

using iree::vm::MakeNativeFunction;

using State = AsyncTestModuleState;

static const vm::NativeFunction<State> kAsyncTestModuleFunctions[] = {
    MakeNativeFunction("return.async.scalar", &State::ReturnAsyncScalar),
};

//===----------------------------------------------------------------------===//
// AsyncTest module instance that will be allocated and reused across contexts
//===----------------------------------------------------------------------===//

class AsyncTestModule final : public vm::NativeModule<AsyncTestModuleState> {
 public:
  AsyncTestModule(iree_vm_instance_t *instance,
                  iree_allocator_t host_allocator);

  StatusOr<std::unique_ptr<AsyncTestModuleState>> CreateState(
      iree_allocator_t host_allocator) override;

 private:
  static constexpr uint32_t kVersion = 0;

  using NativeModule = vm::NativeModule<AsyncTestModuleState>;
};

AsyncTestModule::AsyncTestModule(iree_vm_instance_t *instance,
                                 iree_allocator_t host_allocator)
    : NativeModule("asynctest", AsyncTestModule::kVersion, instance,
                   host_allocator, {kAsyncTestModuleFunctions}) {}

StatusOr<std::unique_ptr<AsyncTestModuleState>> AsyncTestModule::CreateState(
    iree_allocator_t host_allocator) {
  return std::make_unique<AsyncTestModuleState>();
}

}  // namespace openxla::runtime::asynctest

//===----------------------------------------------------------------------===//
// Static IREE VM module registration
//===----------------------------------------------------------------------===//

using namespace openxla::runtime::asynctest;

extern "C" iree_status_t openxla_async_test_module_create(
    iree_vm_instance_t *instance, iree_allocator_t host_allocator,
    iree_vm_module_t **out_module) {
  IREE_ASSERT_ARGUMENT(out_module);

  auto module = std::make_unique<AsyncTestModule>(instance, host_allocator);
  *out_module = module.release()->interface();

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Dynamic IREE VM module registration
//===----------------------------------------------------------------------===//

extern "C" IREE_VM_DYNAMIC_MODULE_EXPORT iree_status_t
openxla_create_async_test_module(iree_vm_dynamic_module_version_t max_version,
                                 iree_vm_instance_t *instance,
                                 iree_host_size_t param_count,
                                 const iree_string_pair_t *params,
                                 iree_allocator_t host_allocator,
                                 iree_vm_module_t **out_module) {
  // Ensure the version matches; the version will change if the VM module
  // interface changes and existing libraries are incompatible.
  if (max_version != IREE_VM_DYNAMIC_MODULE_VERSION_LATEST) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "unsupported runtime version %u, module compiled with version %u",
        max_version, IREE_VM_DYNAMIC_MODULE_VERSION_LATEST);
  }

#if IREE_TRACING_FEATURES
  // Today Tracy cannot be used with custom dynamic modules as it'll try to
  // create a new tracing context distinct from the hosting application. Custom
  // module libraries should be built with tracing disabled.
  fprintf(stderr,
          "Tracy is not currently supported in custom dynamic modules\n");
#endif  // IREE_TRACING_FEATURES

  return openxla_async_test_module_create(instance, host_allocator, out_module);
}
