#pragma once

#include <VkBootstrap.h>
#include <queue>
#include <functional>
#include <vector>

#include <vma/vk_mem_alloc.h>

#include <vk_descriptors.h>
#include <vk_pipelines.h>

struct GLFWwindow;

struct AllocatedImage 
{
	VkImage image;
	VkImageView imageView;
	VmaAllocation allocation;
	VkExtent3D imageExtent;
	VkFormat imageFormat;
};

struct DeletionQueue 
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function);
	void flush();
};

struct FrameData
{
	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;
	DeletionQueue _deletionQueue;

	VkSemaphore _swapchainSemaphore, _renderSemaphore;
	VkFence _renderFence;
};

constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine
{
public:
	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device; 
	VkSurfaceKHR _surface;
	VkAllocationCallbacks* _callBacks;

	VkPipeline _gradientPipeline;
	VkPipelineLayout _gradientPipelineLayout;

	DescriptorAllocator globalDescriptorAllocator;

	VkDescriptorSet _drawImageDescriptors;
	VkDescriptorSetLayout _drawImageDescriptorLayout;
	
	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;

	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	void init();
	void draw();
	void cleanup();
	void draw_background(VkCommandBuffer cmd);

	unsigned int _frameNumber = 0;

	GLFWwindow* get_window() { return window; }

	FrameData _frames[FRAME_OVERLAP];
	FrameData& get_current_frame() {return _frames[_frameNumber % FRAME_OVERLAP]; };

	VkQueue _graphicsQueue;
	VkCommandPoolCreateInfo command_pool_create_info(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags);
	VkCommandBufferAllocateInfo command_buffer_allocate_info(VkCommandPool pool, uint32_t count);

	VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags = 0);
	VkSemaphoreCreateInfo semaphore_create_info(VkSemaphoreCreateFlags flags = 0);

	VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags);
	VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspectMask);

	VkSemaphoreSubmitInfo semaphore_submit_info(VkPipelineStageFlags2 stageMask, VkSemaphore semaphore);
	VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer cmd);
	VkSubmitInfo2 submit_info(VkCommandBufferSubmitInfo* cmd, VkSemaphoreSubmitInfo* signalSemaphoreInfo, VkSemaphoreSubmitInfo* waitSemaaphoreInfo);

	VkImageCreateInfo image_create_info(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent);
	VkImageViewCreateInfo imageview_create_info(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags);

	uint32_t _graphicsQueueFamily;

	DeletionQueue _mainDeletionQueue;

	VmaAllocator _allocator;

	AllocatedImage _drawImage;
	VkExtent2D _drawExtent;
private:
	bool _isInitilized = false;

	void init_vulkan();
	void init_swapchain();
	void init_command();
	void init_sync_structures();

	void init_descriptors();

	void init_pipelines();
	void init_background_pipelines();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();

	GLFWwindow* window;
};

extern VulkanEngine vkEngine;
