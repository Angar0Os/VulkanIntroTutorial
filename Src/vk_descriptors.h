#pragma once

#include <vector>
#include <vulkan/vulkan.h>

#include <span>

struct DescriptorAllocator
{
	struct PoolSizeRatio
	{
		VkDescriptorType type;
		float ratio;
	};

	VkDescriptorPool pool;

	void init_pool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios);
	void clear_descriptors(VkDevice device);
	void destroy_pool(VkDevice device);

	VkDescriptorSet allocate(VkDevice device, VkDescriptorSetLayout layout);
};

struct DescriptorLayoutBuilder
{
	std::vector<VkDescriptorSetLayoutBinding> bindings;

	void add_binding(uint32_t binding, VkDescriptorType type);
	void clear();

	VkDescriptorSetLayout build(VkDevice device, VkShaderStageFlags shaderFlags, void* pNext = nullptr, VkDescriptorSetLayoutCreateFlags flags = 0);
};