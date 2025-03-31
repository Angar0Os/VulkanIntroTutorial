#pragma once

#include <VkBootstrap.h>
#include <queue>
#include <functional>
#include <vector>

#include <vma/vk_mem_alloc.h>

#include <vk_types.h>

#include <vk_descriptors.h>
#include <vk_pipelines.h>

#include <glm/glm.hpp>

struct GLFWwindow;

struct AllocatedImage 
{
	VkImage image;
	VkImageView imageView;
	VmaAllocation allocation;
	VkExtent3D imageExtent;
	VkFormat imageFormat;
};

struct ComputePushConstants
{
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct ComputeEffect 
{
	const char* name;

	VkPipeline pipeline;
	VkPipelineLayout layout;

	ComputePushConstants data;
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
	VkPipelineLayout _trianglePipelineLayout;
	VkPipeline _trianglePipeline;

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
	void draw_geometry(VkCommandBuffer cmd);
	void draw_background(VkCommandBuffer cmd);

	AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags, VmaMemoryUsage memoryUsage);
	void destroy_buffer(const AllocatedBuffer& buffer);

	GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
	void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);

	unsigned int _frameNumber = 0;

	GLFWwindow* get_window() { return window; }

	VkFence _immFence;
	VkCommandBuffer _immCommandBuffer;
	VkCommandPool _immCommandPool;


	FrameData _frames[FRAME_OVERLAP];
	FrameData& get_current_frame() {return _frames[_frameNumber % FRAME_OVERLAP]; };

	VkQueue _graphicsQueue;

	uint32_t _graphicsQueueFamily;

	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{ 0 };

	DeletionQueue _mainDeletionQueue;

	VmaAllocator _allocator;

	AllocatedImage _drawImage;
	VkExtent2D _drawExtent;

	VkPipelineLayout _meshPipelineLayout;
	VkPipeline _meshPipeline;

	GPUMeshBuffers rectangle;
private:
	bool _isInitilized = false;

	void init_vulkan();
	void init_swapchain();
	void init_command();
	void init_sync_structures();

	void init_descriptors();

	void init_pipelines();
	void init_triangle_pipeline();
	void init_background_pipelines();
	void init_mesh_pipeline();
	void init_default_data();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();

	GLFWwindow* window;
};

extern VulkanEngine vkEngine;
