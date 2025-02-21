// CSG_Editor.cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>

//#define CGAL_USE_BASIC_VIEWER
#define GLM_ENABLE_EXPERIMENTAL

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuizmo.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/matrix_inverse.hpp >
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
//#include <CGAL/draw_surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
//#include <CGAL/Polygon_mesh_processing/boolean_operations.h>
#include <CGAL/Polygon_mesh_processing/clip.h>
#include <CGAL/Bbox_3.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>

#include <unordered_map>
#include <optional>

#include "Camera.h"

// CGAL типы
typedef CGAL::Simple_cartesian<double> Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3> CGAL_Mesh;
namespace PMP = CGAL::Polygon_mesh_processing;

glm::vec3 lightPosition(4);
glm::vec3 lightColor(1);
//glm::vec3 viewPosison(5);
bool draw_debug_normals = false;
static Camera camera(glm::vec3(0.0f, 0.0f, 7.0f));

// Шейдеры (встроенные в код для простоты)
static const char* default_vs = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

void main()
{
    FragPos = vec3(uModel * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(uModel))) * aNormal;  
    
    gl_Position = uProj * uView * vec4(FragPos, 1.0);
}
        )";

static const char* default_fs = R"(
#version 330 core
out vec4 FragColor;

in vec3 Normal;  
in vec3 FragPos;  
  
uniform vec3 lightPos; 
uniform vec3 viewPos; 
uniform vec3 lightColor;
uniform vec3 uObjColor;
uniform int facesDirection;

void main()
{
    // ambient
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm = normalize(facesDirection * Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;  
        
    vec3 result = (ambient + diffuse + specular) * uObjColor;
    FragColor = vec4(result, 1.0);
} 
        )";

static const char* debug_normals_fs = R"(
#version 420 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0, 1.0, 0.0, 1.0);
}  
)";

static const char* debug_normals_vs = R"(
#version 420 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out VS_OUT {
    vec3 normal;
} vs_out;

uniform mat4 uView;
uniform mat4 uProj;
uniform mat4 uModel;

void main()
{
    gl_Position = uProj * uView * uModel * vec4(aPos, 1.0); 
    mat3 normalMatrix = mat3(transpose(inverse(uView * uModel)));
    vs_out.normal = normalize(vec3(uProj * vec4(normalMatrix * aNormal, 0.0)));
}
)";

static const char* debug_normals_gs = R"(
#version 420 core
layout (triangles) in;
layout (line_strip, max_vertices = 6) out;

in VS_OUT {
    vec3 normal;
} gs_in[];

const float MAGNITUDE = 0.515;

void GenerateLine(int index)
{
    gl_Position = gl_in[index].gl_Position;
    EmitVertex();
    gl_Position = gl_in[index].gl_Position + vec4(gs_in[index].normal, 0.0) * MAGNITUDE;
    EmitVertex();
    EndPrimitive();
}

void main()
{
    GenerateLine(0);
    GenerateLine(1);
    GenerateLine(2);
}
)";

static void checkCompileErrors(GLuint shader, std::string type)
{
    GLint success;
    GLchar infoLog[1024];
    if (type != "PROGRAM")
    {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
    else
    {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}

// Компиляция шейдеров
static GLuint CompileShader(const char* vertexShader, const char* fragmentShader, const char* geometryShader = nullptr) {
    GLuint vShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vShader, 1, &vertexShader, NULL);
    glCompileShader(vShader);
    checkCompileErrors(vShader, "VERTEX");

    GLuint fShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fShader, 1, &fragmentShader, NULL);
    glCompileShader(fShader);
    checkCompileErrors(fShader, "FRAGMENT");

    unsigned int gShader;
    if (geometryShader != nullptr)
    {
        gShader = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(gShader, 1, &geometryShader, NULL);
        glCompileShader(gShader);
        checkCompileErrors(gShader, "GEOMETRY");
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vShader);
    glAttachShader(prog, fShader);
    if (geometryShader != nullptr)
        glAttachShader(prog, gShader);
    glLinkProgram(prog);

    return prog;
};

// Класс слоя сцены
class SceneLayer {
public:
    std::string name = "Layer";
    bool visible = true;
    bool locked = false;
    ImColor color = ImColor(255, 255, 255);

    SceneLayer(const std::string& name) : name(name) {}
};

// Класс меша с интеграцией CGAL и OpenGL
class Mesh {
public:
    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;
        glm::vec2 texCoord;
    };

    CGAL_Mesh cgal_mesh;
    GLuint vao = 0, vbo = 0, ebo = 0;
    glm::mat4 transform = glm::mat4(1.0f);
    glm::vec3 color = glm::vec3(1.0f);
    bool visible = true;
    int facesDirection = 1;

    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texCoords;

    void flipNormals()
    {
        facesDirection *= -1;
    }

    void generateNormals() {
        normals.clear();
        normals.resize(cgal_mesh.number_of_faces(), glm::vec3(0));
        using Vertexx = CGAL_Mesh::Vertex_index;
        // 1. Вычисляем нормали для каждой грани
        for (auto face : cgal_mesh.faces()) {
            // Получаем вершины грани
            std::vector<Vertexx> face_vertices;
            
            for (auto v : cgal_mesh.vertices_around_face(cgal_mesh.halfedge(face))) {
                face_vertices.push_back(v);
            }

            // Вычисляем нормаль грани
            Kernel::Point_3 p0 = cgal_mesh.point(face_vertices[0]);
            Kernel::Point_3 p1 = cgal_mesh.point(face_vertices[1]);
            Kernel::Point_3 p2 = cgal_mesh.point(face_vertices[2]);

            glm::vec3 v0(p1.x() - p0.x(), p1.y() - p0.y(), p1.z() - p0.z());
            glm::vec3 v1(p2.x() - p0.x(), p2.y() - p0.y(), p2.z() - p0.z());
            glm::vec3 face_normal = glm::normalize(glm::cross(v0, v1));

            std::cout << "Normal: " << "(" << face_normal.x << ", " << face_normal.y << ", " << face_normal.z << ")\n";
            normals[face] = face_normal;
        }
    }

    void uploadToGPU() {
        generateNormals();

        if (!CGAL::is_triangle_mesh(cgal_mesh))
            std::cout << "Input mesh is not triangulated." << std::endl;
        else
            std::cout << "Input mesh is triangulated." << std::endl;

        std::cout << "number_of_faces: " << cgal_mesh.number_of_faces() << std::endl;
        std::cout << "number_of_vertices: " << cgal_mesh.number_of_vertices() << std::endl;

        // 1. Подготовка данных
        std::vector<Vertex> vertices;
        std::vector<GLuint> indices;

        for (auto face : cgal_mesh.faces()) {
            // Получаем вершины грани
            using Vertexx = CGAL_Mesh::Vertex_index;
            std::vector<Vertexx> face_vertices;
            for (auto v : cgal_mesh.vertices_around_face(cgal_mesh.halfedge(face))) {
                face_vertices.push_back(v);
            }

            // Вычисляем нормаль грани
            Kernel::Point_3 p0 = cgal_mesh.point(face_vertices[0]);
            Kernel::Point_3 p1 = cgal_mesh.point(face_vertices[1]);
            Kernel::Point_3 p2 = cgal_mesh.point(face_vertices[2]);
            
            {
                Vertex vertex;
                vertex.position = glm::vec3(p0.x(), p0.y(), p0.z());
                vertex.normal = normals[face];
                vertex.texCoord = glm::vec2();
                vertices.push_back(vertex);
            }

            {
                Vertex vertex;
                vertex.position = glm::vec3(p1.x(), p1.y(), p1.z());
                vertex.normal = normals[face];
                vertex.texCoord = glm::vec2();
                vertices.push_back(vertex);
            }

            {
                Vertex vertex;
                vertex.position = glm::vec3(p2.x(), p2.y(), p2.z());
                vertex.normal = normals[face];
                vertex.texCoord = glm::vec2();
                vertices.push_back(vertex);
            }

        }

        // 2. Создание OpenGL объектов
        if (vao == 0) glGenVertexArrays(1, &vao);
        if (vbo == 0) glGenBuffers(1, &vbo);
        //if (ebo == 0) glGenBuffers(1, &ebo);

        glBindVertexArray(vao);

        // Вершинный буфер
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER,
            vertices.size() * sizeof(Vertex),
            vertices.data(),
            GL_STATIC_DRAW);

        // 3. Настройка атрибутов
        // Позиция
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
            sizeof(Vertex), (void*)offsetof(Vertex, position));

        // Нормаль
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
            sizeof(Vertex), (void*)offsetof(Vertex, normal));

        // UV-координаты
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
            sizeof(Vertex), (void*)offsetof(Vertex, texCoord));

        glBindVertexArray(0);
    }

    // Генерация UV-координат (простая проекция)
    void generateUVs() {
        texCoords.clear();
        for (auto v : cgal_mesh.vertices()) {
            auto p = cgal_mesh.point(v);
            texCoords.emplace_back(p.x(), p.y());
        }
    }

    void render(const glm::mat4& view, const glm::mat4& proj, unsigned int shader_id) const {
        if(vao == 0) return;
        glUniformMatrix4fv(glGetUniformLocation(shader_id, "uView"),
                        1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shader_id, "uProj"),
                        1, GL_FALSE, glm::value_ptr(proj));
        glUniformMatrix4fv(glGetUniformLocation(shader_id, "uModel"),
                        1, GL_FALSE, glm::value_ptr(transform));
        glUniform3fv(glGetUniformLocation(shader_id, "uObjColor"),
                        1, glm::value_ptr(color));
        glUniform3fv(glGetUniformLocation(shader_id, "lightPos"),
            1, glm::value_ptr(lightPosition));
        glUniform3fv(glGetUniformLocation(shader_id, "lightColor"),
            1, glm::value_ptr(lightColor));
        glUniform3fv(glGetUniformLocation(shader_id, "viewPos"),
            1, glm::value_ptr(camera.GetPosition()));
        glUniform1i(glGetUniformLocation(shader_id, "facesDirection"), facesDirection);
        if (facesDirection == 1)
        {
            glCullFace(GL_BACK);
        }
        else if (facesDirection == -1)
        {
            glCullFace(GL_FRONT);
        }
        
        if (vao == 0) return;

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES,0,
            cgal_mesh.num_halfedges());
        glBindVertexArray(0);
    }
};

// Класс объекта сцены
class SceneObject {
public:
    // Добавить в класс SceneObject
    struct Transform {
        glm::vec3 position{ 0.0f };
        glm::quat m_orientation;
        glm::vec3 rotation{ 0.0f }; // В градусах
        glm::vec3 scale{ 1.0f };

        glm::mat4 matrix() const {
            glm::mat4 m = glm::mat4(1.0f);
            m = glm::translate(m, position);
            m = glm::rotate(m, glm::radians(rotation.x), glm::vec3(1, 0, 0));
            m = glm::rotate(m, glm::radians(rotation.y), glm::vec3(0, 1, 0));
            m = glm::rotate(m, glm::radians(rotation.z), glm::vec3(0, 0, 1));
            m = glm::scale(m, scale);
            return m;
        }
    };
public:
    Mesh mesh;
    Transform transform;
    bool selected = false;
    bool isOperationResult = false; // Для автоматически созданных объектов
    int layer = 0;

    // Добавляем AABB для каждого объекта
    CGAL::Bbox_3 bbox;

    void updateBoundingBox() {
        if (mesh.cgal_mesh.is_empty()) return;

        auto bb = CGAL::bbox_3(mesh.cgal_mesh.points().begin(),
            mesh.cgal_mesh.points().end());

        // Применяем трансформацию объекта
        glm::mat4 transform = mesh.transform;
        glm::vec3 min(bb.xmin(), bb.ymin(), bb.zmin());
        glm::vec3 max(bb.xmax(), bb.ymax(), bb.zmax());

        // Трансформируем углы AABB
        std::vector<glm::vec3> corners = {
            glm::vec3(min.x, min.y, min.z),
            glm::vec3(max.x, min.y, min.z),
            glm::vec3(min.x, max.y, min.z),
            glm::vec3(max.x, max.y, min.z),
            glm::vec3(min.x, min.y, max.z),
            glm::vec3(max.x, min.y, max.z),
            glm::vec3(min.x, max.y, max.z),
            glm::vec3(max.x, max.y, max.z)
        };

        glm::vec3 transformedMin(FLT_MAX);
        glm::vec3 transformedMax(-FLT_MAX);

        for (auto& corner : corners) {
            glm::vec3 transformed = transform * glm::vec4(corner, 1.0f);
            transformedMin = glm::min(transformedMin, transformed);
            transformedMax = glm::max(transformedMax, transformed);
        }

        bbox = CGAL::Bbox_3(
            transformedMin.x, transformedMin.y, transformedMin.z,
            transformedMax.x, transformedMax.y, transformedMax.z
        );
    }

    SceneObject(Mesh&& m) : mesh(std::move(m)) {}
    void updateTransform() {
        mesh.transform = transform.matrix();
    }
};

// Класс группы объектов
class ObjectGroup {
public:
    std::string name = "Group";
    std::vector<SceneObject*> objects;
    glm::mat4 transform = glm::mat4(1.0f);

    void applyTransform() {
        for (auto obj : objects) {
            obj->mesh.transform = transform * obj->transform.matrix();
        }
    }

private:
    std::unordered_map<SceneObject*, glm::mat4> localTransforms;

    void captureTransforms() {
        localTransforms.clear();
        const glm::mat4 invGroupTransform = glm::inverse(transform);
        for (auto obj : objects) {
            localTransforms[obj] = invGroupTransform * obj->mesh.transform;
        }
    }
};

// Менеджер сцены
class Scene {
public:
    std::vector<std::unique_ptr<SceneObject>> objects;
    SceneObject* selectedObject = nullptr;
    std::vector<SceneLayer> layers;
    std::vector<ObjectGroup> groups;
    int selectedLayer = 0;

    Scene() {
        layers.emplace_back("Default");
        layers.emplace_back("Walls");
        layers.emplace_back("Lighting");
    }

    SceneObject* addObject(Mesh&& mesh) {
        objects.emplace_back(std::make_unique<SceneObject>(std::move(mesh)));
        auto& obj = objects.back();
        obj->layer = selectedLayer;
        return obj.get();
    }

    void deleteSelected() {
        objects.erase(
            std::remove_if(objects.begin(), objects.end(),
                [this](auto& obj) { return obj.get() == selectedObject; }),
            objects.end()
        );
        selectedObject = nullptr;
    }
};

class Ray {
public:
    glm::vec3 origin;
    glm::vec3 direction;

    struct HitResult {
        float tMin;
        float tMax;
    };

    bool intersects(const CGAL::Bbox_3& bbox, HitResult* result = nullptr) const {
        // Алгоритм Kay-Kajiya для пересечения луча и AABB
        float tmin = -FLT_MAX, tmax = FLT_MAX;

        for (int axis = 0; axis < 3; ++axis) {
            float invD = 1.0f / direction[axis];
            float t0 = (bbox.min(axis) - origin[axis]) * invD;
            float t1 = (bbox.max(axis) - origin[axis]) * invD;

            if (invD < 0.0f) std::swap(t0, t1);

            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;

            if (tmax < tmin) return false;
        }

        if (result) {
            result->tMin = tmin;
            result->tMax = tmax;
        }
        return true;
    }
};

// Обновленная секция CSG операций
class CSGOperations {
public:
    enum class Operaion
    {
        UNION = 0,
        DIFFERENCE,
        INTERSECTION,

    };
    static bool run(Mesh& A, const Mesh& B, Operaion operation) {
        try {
            // Создаем копии мешей для операции
            CGAL_Mesh meshA = A.cgal_mesh;
            CGAL_Mesh meshB = B.cgal_mesh;

            // Применяем трансформации
            applyTransform(meshA, A.transform);
            applyTransform(meshB, B.transform);

            // Выполняем булеву операцию
            bool success = false;

            CGAL_Mesh result;
            switch (operation)
            {
            case CSGOperations::Operaion::UNION:
                 success = PMP::corefine_and_compute_union(
                    meshA,
                    meshB,
                    result,
                    PMP::parameters::default_values(),
                    PMP::parameters::default_values(),
                    PMP::parameters::default_values()
                );
                break;
            case CSGOperations::Operaion::DIFFERENCE:
                 success = PMP::corefine_and_compute_difference(
                    meshA,
                    meshB,
                    result,
                    PMP::parameters::default_values(),
                    PMP::parameters::default_values(),
                    PMP::parameters::default_values()
                );
                break;
            case CSGOperations::Operaion::INTERSECTION:
                 success = PMP::corefine_and_compute_intersection(
                    meshA,
                    meshB,
                    result,
                    PMP::parameters::default_values(),
                    PMP::parameters::default_values(),
                    PMP::parameters::default_values()
                );
                break;
            default:
                break;
            }

            if (success) {
                PMP::remove_degenerate_faces(result);
                A.cgal_mesh = std::move(result);
                applyTransform(A.cgal_mesh, glm::inverse(A.transform));
                A.uploadToGPU();
                return true;
            }
        }
        catch (const std::exception& e) {
            std::cerr << "CSG Error: " << e.what() << "\n";
        }
        return false;
    }

private:
    static void applyTransform(CGAL_Mesh& mesh, const glm::mat4& transform) {
        for (auto v : mesh.vertices()) {
            auto p = mesh.point(v);
            glm::vec4 pos(p.x(), p.y(), p.z(), 1.0f);
            pos = transform * pos;
            mesh.point(v) = Kernel::Point_3(pos.x, pos.y, pos.z);
        }
    }
};

void handleObjectSelection(Scene& scene, GLFWwindow* window,
    const glm::mat4& view, const glm::mat4& projection) {
    if (ImGui::IsMouseClicked(0) && !ImGuizmo::IsUsing() && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
        // Получаем позицию курсора в нормализованных координатах
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);

        int width, height;
        glfwGetWindowSize(window, &width, &height);

        // Преобразуем в координаты NDC
        float x = (2.0f * mouseX) / width - 1.0f;
        float y = 1.0f - (2.0f * mouseY) / height;

        // Создаем луч из камеры
        glm::mat4 invViewProj = glm::inverse(projection * view);
        glm::vec4 rayStart = invViewProj * glm::vec4(x, y, -1.0f, 1.0f);
        glm::vec4 rayEnd = invViewProj * glm::vec4(x, y, 1.0f, 1.0f);

        rayStart /= rayStart.w;
        rayEnd /= rayEnd.w;

        Ray ray;
        ray.origin = glm::vec3(rayStart);
        ray.direction = glm::normalize(glm::vec3(rayEnd - rayStart));

        // Поиск пересечений
        SceneObject* closestObject = nullptr;
        float closestT = FLT_MAX;

        for (auto& obj : scene.objects) {
            obj->updateBoundingBox();

            Ray::HitResult hit;
            if (ray.intersects(obj->bbox, &hit)) {
                if (hit.tMin < closestT && hit.tMin > 0.0f) {
                    closestT = hit.tMin;
                    closestObject = obj.get();
                }
            }
        }

        scene.selectedObject = closestObject;

        // Отладочный вывод
        if (closestObject) {
            std::cout << "Selected object at t = " << closestT
                << ", BBox: " << closestObject->bbox << "\n";
        }
    }
}

// Окно управления слоями
void drawLayerManager(Scene& scene) {
    ImGui::Begin("Layer Manager");

    // Список слоёв
    for (size_t i = 0; i < scene.layers.size(); ++i) {
        auto& layer = scene.layers[i];
        ImGui::PushID(i);

        // Выбор слоя
        if (ImGui::Selectable("##select", scene.selectedLayer == i)) {
            scene.selectedLayer = i;
        }
        ImGui::SameLine();

        // Видимость
        bool visible = layer.visible;
        if (ImGui::Checkbox("##visible", &visible)) {
            layer.visible = visible;
        }
        ImGui::SameLine();

        // Цвет
        ImGui::ColorEdit3("##color", (float*)&layer.color,
            ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
        ImGui::SameLine();

        // Название
        char buf[64];
        strcpy(buf, layer.name.c_str());
        if (ImGui::InputText("##name", buf, sizeof(buf))) {
            layer.name = buf;
        }

        ImGui::PopID();
    }

    // Управление слоями
    if (ImGui::Button("Add Layer")) {
        scene.layers.emplace_back("New Layer");
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove Layer") && scene.layers.size() > 1) {
        scene.layers.erase(scene.layers.begin() + scene.selectedLayer);
        scene.selectedLayer = std::max(0, scene.selectedLayer - 1);
    }

    ImGui::End();
}

// Новое окно свойств
void drawPropertiesWindow(Scene& scene) {
    ImGui::Begin("Object Properties");

    if (scene.selectedObject) {
        SceneObject& obj = *scene.selectedObject;

        // Позиция
        if (ImGui::DragFloat3("Position", glm::value_ptr(obj.transform.position), 0.1f)) {
            obj.updateTransform();
        }

        // Поворот
        glm::vec3 rotation = obj.transform.rotation;
        if (ImGui::DragFloat3("Rotation", glm::value_ptr(rotation), 1.0f, -180.0f, 180.0f)) {
            obj.transform.rotation = rotation;
            obj.updateTransform();
        }

        // Масштаб
        if (ImGui::DragFloat3("Scale", glm::value_ptr(obj.transform.scale), 0.1f, 0.01f, 10.0f)) {
            obj.updateTransform();
        }

        // Цвет
        ImGui::ColorEdit3("Color", glm::value_ptr(obj.mesh.color));

        // Дополнительные параметры
        ImGui::Separator();
        if (ImGui::Button("Reset Transform")) {
            obj.transform = SceneObject::Transform();
            obj.updateTransform();
        }

        ImGui::SameLine();
        ImGui::Checkbox("Visible", &obj.mesh.visible);
        if(ImGui::Button("Flip normals"))
        {
            obj.mesh.flipNormals();
        }

        // Выбор слоя
        int layer = scene.selectedObject->layer;
        if (ImGui::Combo("Layer", &layer,
            [](void* data, int idx, const char** out_text) {
                auto& layers = *static_cast<std::vector<SceneLayer>*>(data);
                *out_text = layers[idx].name.c_str();
                return true;
            }, &scene.layers, scene.layers.size()))
        {
            scene.selectedObject->layer = layer;
        }

        // Управление группами
        if (ImGui::TreeNode("Grouping")) {
            static char groupName[64] = "NewGroup";
            ImGui::InputText("Group Name", groupName, sizeof(groupName));

            if (ImGui::Button("Create Group")) {
                scene.groups.emplace_back();
                scene.groups.back().name = groupName;
            }

            // Присоединение к группе
            for (auto& group : scene.groups) {
                bool inGroup = std::find(group.objects.begin(), group.objects.end(),
                    scene.selectedObject) != group.objects.end();

                if (ImGui::Checkbox(group.name.c_str(), &inGroup)) {
                    if (inGroup) group.objects.push_back(scene.selectedObject);
                    else group.objects.erase(std::remove(group.objects.begin(),
                        group.objects.end(), scene.selectedObject), group.objects.end());
                }
            }
            ImGui::TreePop();
        }
    }
    else {
        ImGui::Text("Select an object to edit properties");
    }

    ImGui::End();
}

static float Width = 1280;
static float Height = 720;
static float zNear = 0.1f;
static float zFar = 500.f;
static float mouse_x = 0.0;
static float mouse_y = 0.0;
static float lastX = 0.0;
static float lastY = 0.0;
static bool keys[512];
static bool dev_mode = false;

void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    if (action == GLFW_PRESS)
        keys[key] = true;
    else if (action == GLFW_RELEASE)
        keys[key] = false;

    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        case GLFW_KEY_F1:
            dev_mode = !dev_mode;
            glfwSetInputMode(window, GLFW_CURSOR, dev_mode ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
        default:;
        }
    }
}

void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
    {
        double xPos;
        double yPos;
        //getting cursor position
        glfwGetCursorPos(window, &xPos, &yPos);

        mouse_x = (static_cast<float>(xPos) - Width / 2.f) / Width * 2.f;
        mouse_y = -(static_cast<float>(yPos) - Height / 2.f) / Height * 2.f;
    }
}

static void MouseCallback(GLFWwindow* window, double xpos, double ypos)
{
    static bool firstMouse = true;
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    const float xoffset = xpos - lastX;
    const float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    if (dev_mode)
        return;
    camera.ProcessMouseMovement(xoffset, yoffset);
}

static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}

// Инициализация GLFW
GLFWwindow* initGLFW() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    GLFWwindow* window = glfwCreateWindow(Width, Height, "CSG Editor", NULL, NULL);
    glfwMakeContextCurrent(window);

    glfwSetMouseButtonCallback(window, MouseButtonCallback);
    glfwSetCursorPosCallback(window, MouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetScrollCallback(window, ScrollCallback);
    glfwSetKeyCallback(window, KeyCallback);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);
    return window;
}

// Инициализация ImGui
void initImGui(GLFWwindow* window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); 
    io.ConfigFlags |= ImGuiConfigFlags_::ImGuiConfigFlags_DockingEnable;
    
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 460");
    ImGui::StyleColorsDark();
    
    ImGuizmo::SetImGuiContext(ImGui::GetCurrentContext());
}

// Создание куба
void createCube(Mesh& mesh, float size = 1.f) {
    using Vertex = CGAL_Mesh::Vertex_index;

    // Вершины куба (8 точек)
    std::vector<Vertex> vertices = {
        mesh.cgal_mesh.add_vertex(Kernel::Point_3(0, 0, 0)),      // 0
        mesh.cgal_mesh.add_vertex(Kernel::Point_3(size, 0, 0)),   // 1
        mesh.cgal_mesh.add_vertex(Kernel::Point_3(size, size, 0)),// 2
        mesh.cgal_mesh.add_vertex(Kernel::Point_3(0, size, 0)),   // 3
        mesh.cgal_mesh.add_vertex(Kernel::Point_3(0, 0, size)),   // 4
        mesh.cgal_mesh.add_vertex(Kernel::Point_3(size, 0, size)),// 5
        mesh.cgal_mesh.add_vertex(Kernel::Point_3(size, size, size)), //6
        mesh.cgal_mesh.add_vertex(Kernel::Point_3(0, size, size)) //7
    };

    // Функция для создания двух треугольников из квада
    auto add_quad = [&](Vertex a, Vertex b, Vertex c, Vertex d) {
        mesh.cgal_mesh.add_face(a, b, c);
        mesh.cgal_mesh.add_face(a, c, d);
        };

    // Грани куба (каждая грань - два треугольника)
    add_quad(vertices[3], vertices[2], vertices[1], vertices[0]); // Низ
    add_quad(vertices[4], vertices[5], vertices[6], vertices[7]); // Верх
    add_quad(vertices[0], vertices[1], vertices[5], vertices[4]); // Перед
    add_quad(vertices[2], vertices[3], vertices[7], vertices[6]); // Зад
    add_quad(vertices[1], vertices[2], vertices[6], vertices[5]); // Право
    add_quad(vertices[3], vertices[0], vertices[4], vertices[7]); // Лево
    mesh.uploadToGPU();
}    

// Работа с Gizmo
void handleGizmo(SceneObject& obj, const glm::mat4& view, const glm::mat4& proj) {
    static ImGuizmo::OPERATION op = ImGuizmo::TRANSLATE;
    static ImGuizmo::MODE mode = ImGuizmo::WORLD;

    if (ImGui::IsKeyPressed(ImGuiKey_W)) op = ImGuizmo::TRANSLATE;
    if (ImGui::IsKeyPressed(ImGuiKey_E)) op = ImGuizmo::ROTATE;
    if (ImGui::IsKeyPressed(ImGuiKey_R)) op = ImGuizmo::SCALE;

    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetRect(0, 0,
        ImGui::GetIO().DisplaySize.x,
        ImGui::GetIO().DisplaySize.y);

    glm::mat4 delta;
    ImGuizmo::Manipulate(
        glm::value_ptr(view),
        glm::value_ptr(proj),
        op,
        mode,
        glm::value_ptr(obj.mesh.transform),
        glm::value_ptr(delta)
    );

    // Обновляем Transform из матрицы
    if (ImGuizmo::IsUsing()) {
        glm::vec3 skew;
        glm::vec4 perspective;
        glm::decompose(
            obj.mesh.transform,
            obj.transform.scale,
            obj.transform.m_orientation,
            obj.transform.position,
            skew,
            perspective
        );
        obj.transform.rotation = glm::degrees(glm::eulerAngles(obj.transform.m_orientation));
    }
}

void DoMovement(float dt)
{
    if (dev_mode)
        return;

    // Camera controls
    if (keys[GLFW_KEY_W])
        camera.ProcessKeyboard(Camera::Movement::FORWARD, dt);
    if (keys[GLFW_KEY_S])
        camera.ProcessKeyboard(Camera::Movement::BACKWARD, dt);
    if (keys[GLFW_KEY_A])
        camera.ProcessKeyboard(Camera::Movement::LEFT, dt);
    if (keys[GLFW_KEY_D])
        camera.ProcessKeyboard(Camera::Movement::RIGHT, dt);
    if (keys[GLFW_KEY_SPACE])
        camera.ProcessKeyboard(Camera::Movement::UP, dt);
}

double getCurrentTime()
{
    return glfwGetTime();
}

int CalculateFrameRate(float current)
{
    static double last = 0.0;
    static int FPS = 0;
    ++FPS;
    static int out_FPS = 0;
    if (current - last >= 1.0f)
    {
        out_FPS = FPS;
        FPS = 0;
        last = current;
    }

    return out_FPS;
}

int ChangeDeltaTime(float& deltaTime, float& lastFrame)
{
    auto currentFrame = getCurrentTime();

    auto FPS = CalculateFrameRate(currentFrame);
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    return FPS;
}

int main() {
    GLFWwindow* window = initGLFW();
    initImGui(window);
    glEnable(GL_CULL_FACE);
    // Камера
    /*glm::mat4 view = glm::lookAt(
        viewPosison,
        glm::vec3(0, 0, 0),
        glm::vec3(0, 1, 0)
    );
    glm::mat4 proj = glm::perspective(
        glm::radians(45.0f),
        1280.0f / 720.0f,
        0.1f,
        100.0f
    );*/


    Scene scene;

    auto shader_id = CompileShader(default_vs, default_fs);
    auto debug_normals_shader_id = CompileShader(debug_normals_vs, debug_normals_fs, debug_normals_gs);
    // Рендеринг

// Обновлённый рендеринг с учётом слоёв
    auto renderScene = [&](const glm::mat4& view, const glm::mat4& proj) {
        for (auto& layer : scene.layers) {
            if (!layer.visible) continue;

            for (auto& obj : scene.objects) {
                if (obj->layer != &layer - &scene.layers[0]) continue;
                if (!obj->mesh.visible) continue;

                glUseProgram(shader_id);
                obj->mesh.render(view, proj, shader_id);
                if (draw_debug_normals)
                {
                    glUseProgram(debug_normals_shader_id);
                    obj->mesh.render(view, proj, debug_normals_shader_id);
                }
            }
        }
        };

    {
        Mesh cube1;
        createCube(cube1);
        cube1.flipNormals();
        cube1.color = glm::vec3(0.8f, 0.2f, 0.3f);
        auto obj = scene.addObject(std::move(cube1));
        obj->transform.scale = glm::vec3(3.f);
        obj->transform.position = glm::vec3(-1.f, 0.f, -1.f);
        obj->updateTransform();

        Mesh cube2;
        createCube(cube2);
        cube2.color = glm::vec3(0.2f, 0.8f, 0.3f);
        auto obj2 = scene.addObject(std::move(cube2));
        obj2->transform.scale = glm::vec3(1.f, 3.f, 1.f);
        obj2->updateTransform();

        scene.selectedObject = obj2;
        CSGOperations::run(obj->mesh, scene.selectedObject->mesh, CSGOperations::Operaion::DIFFERENCE);
        scene.deleteSelected();

        Mesh cube3;
        cube3.flipNormals();
        createCube(cube3);
        cube3.color = glm::vec3(0.6f, 0.8f, 0.3f);
        auto obj3 = scene.addObject(std::move(cube3));
        obj3->transform.position = glm::vec3(0.1f, 1.f, -2.f);
        obj3->transform.scale = glm::vec3(0.8f, 0.8f, 5.f);
        obj3->updateTransform();

        scene.selectedObject = obj3;
        CSGOperations::run(obj->mesh, scene.selectedObject->mesh, CSGOperations::Operaion::UNION);
        scene.deleteSelected();
    }
    float dt = 0.f;
    float lastFrame = getCurrentTime();
    // Главный цикл
    while (!glfwWindowShouldClose(window)) {
        ChangeDeltaTime(dt, lastFrame);
        glClearColor(0.4f, 0.4f, 0.4f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        DoMovement(dt);

        // Обновление ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();

        glm::mat4 proj = glm::perspective(glm::radians(camera.GetZoom()), static_cast<float>(Width) / Height, zNear, zFar);
        glm::mat4 view = camera.GetViewMatrix();

        // Окно редактора
        ImGui::Begin("CSG Editor");
        if (ImGui::Button("Add Cube")) {
            Mesh cube;
            createCube(cube);
            cube.uploadToGPU();
            cube.color = glm::vec3(0.8f, 0.2f, 0.3f);
            scene.addObject(std::move(cube));
        }
        if (ImGui::Button("Polygon Mode line"))
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }
        if (ImGui::Button("Polygon Mode fill"))
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
        if (ImGui::Button("Draw debug normals"))
        {
            draw_debug_normals = !draw_debug_normals;
        }
        ImGui::End();

        // Окно свойств
        drawPropertiesWindow(scene);
        drawLayerManager(scene);
        //drawGroupManager(scene); // Аналогично layer manager

        // Кнопки CSG
        ImGui::Begin("CSG Tools");
        auto run_button = [&](CSGOperations::Operaion operation)
            {
                // Для демо - вычитаем первый попавшийся объект
                for (auto& obj : scene.objects) {
                    if (obj.get() != scene.selectedObject && !obj->isOperationResult) {
                        if (CSGOperations::run(obj.get()->mesh, scene.selectedObject->mesh, operation)) {
                            scene.deleteSelected(); // Удаляем оригиналы
                            break;
                        }
                    }
                }
            };

        if (ImGui::Button("Difference") && scene.selectedObject) {
            run_button(CSGOperations::Operaion::DIFFERENCE);
        }
        if (ImGui::Button("Union") && scene.selectedObject) {
            run_button(CSGOperations::Operaion::UNION);
        }
        if (ImGui::Button("Intersection") && scene.selectedObject) {
            run_button(CSGOperations::Operaion::INTERSECTION);
        }
        ImGui::End();

        // Рендеринг

        if (scene.selectedObject) {
            handleGizmo(*scene.selectedObject, view, proj);
        }

        // 2. Проверка: если UI поглотил ввод, пропускаем обработку сцены
        bool uiWantsInput = ImGui::GetIO().WantCaptureMouse || ImGuizmo::IsUsing();
        if (!uiWantsInput) {
            handleObjectSelection(scene, window, view, proj);
        }

        renderScene(view, proj);

        // Рендеринг ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}