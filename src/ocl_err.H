
char const * get_cl_err_str( cl_int const & err ) {
  if(0){}
  else if( err == CL_SUCCESS) { return "CL_SUCCESS"; }
  else if( err == CL_DEVICE_NOT_FOUND) { return "CL_DEVICE_NOT_FOUND"; }
  else if( err == CL_DEVICE_NOT_AVAILABLE) { return "CL_DEVICE_NOT_AVAILABLE"; }
  else if( err == CL_COMPILER_NOT_AVAILABLE) { return "CL_COMPILER_NOT_AVAILABLE"; }
  else if( err == CL_MEM_OBJECT_ALLOCATION_FAILURE) { return "CL_MEM_OBJECT_ALLOCATION_FAILURE"; }
  else if( err == CL_OUT_OF_RESOURCES) { return "CL_OUT_OF_RESOURCES"; }
  else if( err == CL_OUT_OF_HOST_MEMORY) { return "CL_OUT_OF_HOST_MEMORY"; }
  else if( err == CL_PROFILING_INFO_NOT_AVAILABLE) { return "CL_PROFILING_INFO_NOT_AVAILABLE"; }
  else if( err == CL_MEM_COPY_OVERLAP) { return "CL_MEM_COPY_OVERLAP"; }
  else if( err == CL_IMAGE_FORMAT_MISMATCH) { return "CL_IMAGE_FORMAT_MISMATCH"; }
  else if( err == CL_IMAGE_FORMAT_NOT_SUPPORTED) { return "CL_IMAGE_FORMAT_NOT_SUPPORTED"; }
  else if( err == CL_BUILD_PROGRAM_FAILURE) { return "CL_BUILD_PROGRAM_FAILURE"; }
  else if( err == CL_MAP_FAILURE) { return "CL_MAP_FAILURE"; }
  else if( err == CL_MISALIGNED_SUB_BUFFER_OFFSET) { return "CL_MISALIGNED_SUB_BUFFER_OFFSET"; }
  else if( err == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST) { return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; }
//  else if( err == CL_COMPILE_PROGRAM_FAILURE) { return "CL_COMPILE_PROGRAM_FAILURE"; }
//  else if( err == CL_LINKER_NOT_AVAILABLE) { return "CL_LINKER_NOT_AVAILABLE"; }
//  else if( err == CL_LINK_PROGRAM_FAILURE) { return "CL_LINK_PROGRAM_FAILURE"; }
//  else if( err == CL_DEVICE_PARTITION_FAILED) { return "CL_DEVICE_PARTITION_FAILED"; }
//  else if( err == CL_KERNEL_ARG_INFO_NOT_AVAILABLE) { return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; }
  else if( err == CL_INVALID_VALUE) { return "CL_INVALID_VALUE"; }
  else if( err == CL_INVALID_DEVICE_TYPE) { return "CL_INVALID_DEVICE_TYPE"; }
  else if( err == CL_INVALID_PLATFORM) { return "CL_INVALID_PLATFORM"; }
  else if( err == CL_INVALID_DEVICE) { return "CL_INVALID_DEVICE"; }
  else if( err == CL_INVALID_CONTEXT) { return "CL_INVALID_CONTEXT"; }
  else if( err == CL_INVALID_QUEUE_PROPERTIES) { return "CL_INVALID_QUEUE_PROPERTIES"; }
  else if( err == CL_INVALID_COMMAND_QUEUE) { return "CL_INVALID_COMMAND_QUEUE"; }
  else if( err == CL_INVALID_HOST_PTR) { return "CL_INVALID_HOST_PTR"; }
  else if( err == CL_INVALID_MEM_OBJECT) { return "CL_INVALID_MEM_OBJECT"; }
  else if( err == CL_INVALID_IMAGE_FORMAT_DESCRIPTOR) { return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; }
  else if( err == CL_INVALID_IMAGE_SIZE) { return "CL_INVALID_IMAGE_SIZE"; }
  else if( err == CL_INVALID_SAMPLER) { return "CL_INVALID_SAMPLER"; }
  else if( err == CL_INVALID_BINARY) { return "CL_INVALID_BINARY"; }
  else if( err == CL_INVALID_BUILD_OPTIONS) { return "CL_INVALID_BUILD_OPTIONS"; }
  else if( err == CL_INVALID_PROGRAM) { return "CL_INVALID_PROGRAM"; }
  else if( err == CL_INVALID_PROGRAM_EXECUTABLE) { return "CL_INVALID_PROGRAM_EXECUTABLE"; }
  else if( err == CL_INVALID_KERNEL_NAME) { return "CL_INVALID_KERNEL_NAME"; }
  else if( err == CL_INVALID_KERNEL_DEFINITION) { return "CL_INVALID_KERNEL_DEFINITION"; }
  else if( err == CL_INVALID_KERNEL) { return "CL_INVALID_KERNEL"; }
  else if( err == CL_INVALID_ARG_INDEX) { return "CL_INVALID_ARG_INDEX"; }
  else if( err == CL_INVALID_ARG_VALUE) { return "CL_INVALID_ARG_VALUE"; }
  else if( err == CL_INVALID_ARG_SIZE) { return "CL_INVALID_ARG_SIZE"; }
  else if( err == CL_INVALID_KERNEL_ARGS) { return "CL_INVALID_KERNEL_ARGS"; }
  else if( err == CL_INVALID_WORK_DIMENSION) { return "CL_INVALID_WORK_DIMENSION"; }
  else if( err == CL_INVALID_WORK_GROUP_SIZE) { return "CL_INVALID_WORK_GROUP_SIZE"; }
  else if( err == CL_INVALID_WORK_ITEM_SIZE) { return "CL_INVALID_WORK_ITEM_SIZE"; }
  else if( err == CL_INVALID_GLOBAL_OFFSET) { return "CL_INVALID_GLOBAL_OFFSET"; }
  else if( err == CL_INVALID_EVENT_WAIT_LIST) { return "CL_INVALID_EVENT_WAIT_LIST"; }
  else if( err == CL_INVALID_EVENT) { return "CL_INVALID_EVENT"; }
  else if( err == CL_INVALID_OPERATION) { return "CL_INVALID_OPERATION"; }
  else if( err == CL_INVALID_GL_OBJECT) { return "CL_INVALID_GL_OBJECT"; }
  else if( err == CL_INVALID_BUFFER_SIZE) { return "CL_INVALID_BUFFER_SIZE"; }
  else if( err == CL_INVALID_MIP_LEVEL) { return "CL_INVALID_MIP_LEVEL"; }
  else if( err == CL_INVALID_GLOBAL_WORK_SIZE) { return "CL_INVALID_GLOBAL_WORK_SIZE"; }
  else if( err == CL_INVALID_PROPERTY) { return "CL_INVALID_PROPERTY"; }
//  else if( err == CL_INVALID_IMAGE_DESCRIPTOR) { return "CL_INVALID_IMAGE_DESCRIPTOR"; }
//  else if( err == CL_INVALID_COMPILER_OPTIONS) { return "CL_INVALID_COMPILER_OPTIONS"; }
//  else if( err == CL_INVALID_LINKER_OPTIONS) { return "CL_INVALID_LINKER_OPTIONS"; }
//  else if( err == CL_INVALID_DEVICE_PARTITION_COUNT) { return "CL_INVALID_DEVICE_PARTITION_COUNT"; }
  else return "Unknown OpenCL error code";
}

