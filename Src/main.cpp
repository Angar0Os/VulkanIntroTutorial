#include <vk_engine.h>

#include <GLFW/glfw3.h>

int main(int argc, char** argv)
{
	vkEngine.init();	

	do {
		glfwPollEvents();


		vkEngine.draw();
	} while (!glfwWindowShouldClose(vkEngine.get_window()));

	vkEngine.cleanup();
}