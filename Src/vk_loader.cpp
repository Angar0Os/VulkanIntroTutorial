//#include <stb_image.h>
//#include <iostream>
//#include <vk_loader.h>
//
//#include <vk_engine.h>
//#include <vk_initializers.h>
//#include <vk_types.h>
//
//#define GLM_ENABLE_EXPERIMENTAL
//#include <glm/gtx/quaternion.hpp>
//#include <glm/glm.hpp>
//
//#include <include/fastgltf/glm_element_traits.hpp>
//#include <include/fastgltf/tools.hpp>
//#include <include/fastgltf/parser.hpp>
//
//std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath)
//{
//    std::cout << "Loading GLTF: " << filePath << std::endl;
//
//    fastgltf::GltfDataBuffer data;
//    data.loadFromFile(filePath);
//
//    constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers
//        | fastgltf::Options::LoadExternalBuffers;
//
//    fastgltf::Asset gltf;
//    fastgltf::Parser parser{};
//
//    auto load = parser.loadBinaryGLTF(&data, filePath.parent_path(), gltfOptions);
//    if (load) {
//        gltf = std::move(load.get());
//    }
//    else {
//        std::cout << "Failed to load glTF: {} \n" << fastgltf::to_underlying(load.error()) << std::endl;
//        return {};
//    }
//    std::vector<std::shared_ptr<MeshAsset>> meshes;
//
//    std::vector<uint32_t> indices;
//    std::vector<Vertex> vertices;
//    for (fastgltf::Mesh& mesh : gltf.meshes)
//    {
//        MeshAsset newmesh;
//
//        newmesh.name = mesh.name;
//
//        indices.clear();
//        vertices.clear();
//
//        for (auto&& p : mesh.primitives) {
//            GeoSurface newSurface;
//            newSurface.startIndex = (uint32_t)indices.size();
//            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;
//
//            size_t initial_vtx = vertices.size();
//
//            {
//                fastgltf::Accessor& indexaccessor = gltf.accessors[p.indicesAccessor.value()];
//                indices.reserve(indices.size() + indexaccessor.count);
//
//                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexaccessor,
//                    [&](std::uint32_t idx) {
//                        indices.push_back(idx + initial_vtx);
//                    });
//            }
//
//            {
//                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
//                vertices.resize(vertices.size() + posAccessor.count);
//
//                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor,
//                    [&](glm::vec3 v, size_t index) {
//                        Vertex newvtx;
//                        newvtx.position = v;
//                        newvtx.normal = { 1, 0, 0 };
//                        newvtx.color = glm::vec4{ 1.f };
//                        newvtx.uv_x = 0;
//                        newvtx.uv_y = 0;
//                        vertices[initial_vtx + index] = newvtx;
//                    });
//            }
//
//            auto normals = p.findAttribute("NORMAL");
//            if (normals != p.attributes.end()) {
//
//                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[(*normals).second],
//                    [&](glm::vec3 v, size_t index) {
//                        vertices[initial_vtx + index].normal = v;
//                    });
//            }
//
//            auto uv = p.findAttribute("TEXCOORD_0");
//            if (uv != p.attributes.end()) {
//
//                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
//                    [&](glm::vec2 v, size_t index) {
//                        vertices[initial_vtx + index].uv_x = v.x;
//                        vertices[initial_vtx + index].uv_y = v.y;
//                    });
//            }
//
//            auto colors = p.findAttribute("COLOR_0");
//            if (colors != p.attributes.end()) {
//
//                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[(*colors).second],
//                    [&](glm::vec4 v, size_t index) {
//                        vertices[initial_vtx + index].color = v;
//                    });
//            }
//            newmesh.surfaces.push_back(newSurface);
//        }
//
//        // display the vertex normals
//        constexpr bool OverrideColors = true;
//        if (OverrideColors) {
//            for (Vertex& vtx : vertices) {
//                vtx.color = glm::vec4(vtx.normal, 1.f);
//            }
//        }
//        newmesh.meshBuffers = engine->uploadMesh(indices, vertices);
//
//        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newmesh)));
//    }
//
//    return meshes;
//}