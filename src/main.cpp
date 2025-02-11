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
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
//#include <CGAL/draw_surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/corefinement.h>
//#include <CGAL/Polygon_mesh_processing/boolean_operations.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <memory>

#include <unordered_map>
#include <optional>

// CGAL типы
typedef CGAL::Simple_cartesian<double> Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3> CGAL_Mesh;
namespace PMP = CGAL::Polygon_mesh_processing;

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
    CGAL_Mesh cgal_mesh;
    GLuint vao = 0, vbo = 0;
    glm::mat4 transform = glm::mat4(1.0f);
    glm::vec3 color = glm::vec3(1.0f);
    bool visible = true;

    void uploadToGPU() {
        std::vector<float> vertexData;

        if (!CGAL::is_triangle_mesh(cgal_mesh))
            std::cout << "Input mesh is not triangulated." << std::endl;
        else
            std::cout << "Input mesh is triangulated." << std::endl;

        //PMP::triangulate_faces(cgal_mesh);

        std::cout << "number_of_faces: " << cgal_mesh.number_of_faces() << std::endl;
        std::cout << "number_of_vertices: " << cgal_mesh.number_of_vertices() << std::endl;
        
        // Собираем данные вершин
        for(auto f : cgal_mesh.faces()) {
          
            for (auto& v : cgal_mesh.vertices_around_face(cgal_mesh.halfedge(f)))
            {
                auto p = cgal_mesh.point(v);
                //auto p = v;
                vertexData.push_back(p.x());
                vertexData.push_back(p.y());
                vertexData.push_back(p.z());
                // Базовые нормали для плоского затенения
                vertexData.push_back(0.0f);
                vertexData.push_back(1.0f);
                vertexData.push_back(0.0f);

            }
        }

        // Генерация OpenGL объектов
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, 
                    vertexData.size() * sizeof(float),
                    vertexData.data(), 
                    GL_STATIC_DRAW);

        // Атрибуты вершин (позиция + нормаль)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }

    void render(const glm::mat4& viewProj) const {
        if(vao == 0) return;

        // Шейдеры (встроенные в код для простоты)
        static const char* vertexShader = R"(
            #version 460 core
            layout(location = 0) in vec3 aPos;
            layout(location = 1) in vec3 aNormal;
            uniform mat4 uViewProj;
            uniform mat4 uModel;
            out vec3 vNormal;
            void main() {
                gl_Position = uViewProj * uModel * vec4(aPos, 1.0);
                vNormal = mat3(transpose(inverse(uModel))) * aNormal;
            }
        )";

        static const char* fragmentShader = R"(
            #version 460 core
            in vec3 vNormal;
            uniform vec3 uColor;
            out vec4 FragColor;
            void main() {
                vec3 lightDir = normalize(vec3(0.5, 1.0, 0.7));
                float diff = max(dot(vNormal, lightDir), 0.2);
                FragColor = vec4(uColor * diff, 1.0);
            }
        )";

        // Компиляция шейдеров
        static GLuint program = [](){
            GLuint vShader = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(vShader, 1, &vertexShader, NULL);
            glCompileShader(vShader);
            
            GLuint fShader = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(fShader, 1, &fragmentShader, NULL);
            glCompileShader(fShader);
            
            GLuint prog = glCreateProgram();
            glAttachShader(prog, vShader);
            glAttachShader(prog, fShader);
            glLinkProgram(prog);
            return prog;
        }();

        // Рендеринг
        glUseProgram(program);
        glUniformMatrix4fv(glGetUniformLocation(program, "uViewProj"), 
                        1, GL_FALSE, glm::value_ptr(viewProj));
        glUniformMatrix4fv(glGetUniformLocation(program, "uModel"), 
                        1, GL_FALSE, glm::value_ptr(transform));
        glUniform3fv(glGetUniformLocation(program, "uColor"), 
                        1, glm::value_ptr(color));
        
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, cgal_mesh.number_of_vertices() * cgal_mesh.number_of_faces());
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

    void addObject(Mesh&& mesh) {
        objects.emplace_back(std::make_unique<SceneObject>(std::move(mesh)));
        objects.back()->layer = selectedLayer;
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

// Ray casting для выбора объектов
class Ray {
public:
    glm::vec3 origin;
    glm::vec3 direction;

    static Ray fromMouse(GLFWwindow* window, const glm::mat4& view, const glm::mat4& proj) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);

        int width, height;
        glfwGetWindowSize(window, &width, &height);

        glm::vec4 rayClip(
            (2.0f * x) / width - 1.0f,
            1.0f - (2.0f * y) / height,
            -1.0f,
            1.0f
        );

        glm::vec4 rayEye = glm::inverse(proj) * rayClip;
        rayEye = glm::vec4(rayEye.x, rayEye.y, -1.0f, 0.0f);

        glm::vec3 rayWorld = glm::vec3(glm::inverse(view) * rayEye);
        return { glm::vec3(glm::inverse(view)[3]), glm::normalize(rayWorld) };
    }
};

// Обновленная секция CSG операций
class CSGOperations {
public:
    static bool difference(Mesh& A, const Mesh& B) {
        try {
            // Создаем копии мешей для операции
            CGAL_Mesh meshA = A.cgal_mesh;
            CGAL_Mesh meshB = B.cgal_mesh;

            // Применяем трансформации
            applyTransform(meshA, A.transform);
            applyTransform(meshB, B.transform);

            // Выполняем булеву операцию
            CGAL_Mesh result;
            bool success = PMP::corefine_and_compute_difference(
                meshA,
                meshB,
                result,
                PMP::parameters::default_values(),
                PMP::parameters::default_values(),
                PMP::parameters::default_values()
            );

            if (success) {
                A.cgal_mesh = std::move(result);
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


// Прототипы функций
GLFWwindow* initGLFW();
void initImGui(GLFWwindow* window);
void createCube(Mesh& mesh, float size = 1.0f);
void handleGizmo(SceneObject& obj, const glm::mat4& view, const glm::mat4& proj);

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

int main() {
    GLFWwindow* window = initGLFW();
    initImGui(window);
    
    // Камера
    glm::mat4 view = glm::lookAt(
        glm::vec3(5,5,5), 
        glm::vec3(0,0,0), 
        glm::vec3(0,1,0)
    );
    glm::mat4 proj = glm::perspective(
        glm::radians(45.0f), 
        1280.0f/720.0f, 
        0.1f, 
        100.0f
    );

    Scene scene;

// Обновлённый рендеринг с учётом слоёв
    auto renderScene = [&](const glm::mat4& viewProj) {
        for (auto& layer : scene.layers) {
            if (!layer.visible) continue;

            for (auto& obj : scene.objects) {
                if (obj->layer != &layer - &scene.layers[0]) continue;
                if (!obj->mesh.visible) continue;

                obj->mesh.render(viewProj);
            }
        }
        };

    // Главный цикл
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Обновление ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();

        // Обработка выбора объектов
        if (ImGui::IsMouseClicked(0) && ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow)) {
            Ray ray = Ray::fromMouse(window, view, proj);

            float minDist = FLT_MAX;
            for (auto& obj : scene.objects) {
                if (obj->isOperationResult) continue;

                // Простая проверка AABB
                glm::vec3 objPos(obj->mesh.transform[3]);
                float dist = glm::distance(ray.origin, objPos);
                if (dist < minDist) {
                    scene.selectedObject = obj.get();
                    minDist = dist;
                }
            }
        }

        // Окно редактора
        ImGui::Begin("CSG Editor");
        if (ImGui::Button("Add Cube")) {
            Mesh cube;
            createCube(cube);
            cube.uploadToGPU();
            cube.color = glm::vec3(0.8f, 0.2f, 0.3f);
            scene.addObject(std::move(cube));
        }
        ImGui::End();

        // Окно свойств
        drawPropertiesWindow(scene);
        drawLayerManager(scene);
        //drawGroupManager(scene); // Аналогично layer manager

        // Окно вьюпорта
        ImGui::Begin("Viewport", nullptr, 
            ImGuiWindowFlags_NoScrollbar | 
            ImGuiWindowFlags_NoScrollWithMouse);

        // Кнопки CSG
        ImGui::Begin("CSG Tools");
        if (ImGui::Button("Difference") && scene.selectedObject) {
            // Для демо - вычитаем первый попавшийся объект
            for (auto& obj : scene.objects) {
                if (obj.get() != scene.selectedObject && !obj->isOperationResult) {
                    if (CSGOperations::difference(obj.get()->mesh, scene.selectedObject->mesh)) {
                        scene.deleteSelected(); // Удаляем оригиналы
                        break;
                    }
                }
            }
        }
        ImGui::End();

        // Рендеринг
        renderScene(proj * view);
        if (scene.selectedObject) {
            handleGizmo(*scene.selectedObject, view, proj);
        }
        
        ImGui::End();

        // Рендеринг ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

// Инициализация GLFW
GLFWwindow* initGLFW() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    GLFWwindow* window = glfwCreateWindow(1280, 720, "CSG Editor", NULL, NULL);
    glfwMakeContextCurrent(window);
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
void createCube(Mesh& mesh, float size) {
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