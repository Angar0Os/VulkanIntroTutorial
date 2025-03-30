#pragma once

#include <vk_engine.h>


namespace vkutils
{
	bool load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule);
}
