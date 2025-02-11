// CSG_Editor.cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>

//#define CGAL_USE_BASIC_VIEWER

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <ImGuizmo.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
//#include <CGAL/draw_surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <vector>
#include <iostream>

// CGAL типы
typedef CGAL::Simple_cartesian<double> Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3> CGAL_Mesh;

// Класс меша с интеграцией CGAL и OpenGL
class Mesh {
public:
    CGAL_Mesh cgal_mesh;
    GLuint vao = 0, vbo = 0;
    glm::mat4 transform = glm::mat4(1.0f);
    glm::vec3 color = glm::vec3(1.0f);

    void uploadToGPU() {
        std::vector<float> vertexData;

        if (!CGAL::is_triangle_mesh(cgal_mesh))
            std::cout << "Input mesh is not triangulated." << std::endl;
        else
            std::cout << "Input mesh is triangulated." << std::endl;
        namespace PMP = CGAL::Polygon_mesh_processing;
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

// Прототипы функций
GLFWwindow* initGLFW();
void initImGui(GLFWwindow* window);
void createCube(Mesh& mesh, float size = 1.0f);
void handleGizmo(Mesh& mesh, const glm::mat4& view, const glm::mat4& proj);

int main() {
    GLFWwindow* window = initGLFW();
    initImGui(window);
    
    // Тестовый меш
    Mesh cube;
    createCube(cube);
    cube.uploadToGPU();
    cube.color = glm::vec3(0.8f, 0.2f, 0.3f);
    
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

    // Главный цикл
    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Обновление ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();

        // Окно редактора
        ImGui::Begin("CSG Editor");
        if (ImGui::Button("Add Cube")) {
            // Добавление новых объектов
        }
        ImGui::End();

        // Окно вьюпорта
        ImGui::Begin("Viewport", nullptr, 
            ImGuiWindowFlags_NoScrollbar | 
            ImGuiWindowFlags_NoScrollWithMouse);
        
        // Рендеринг 3D
        cube.render(proj * view);
        handleGizmo(cube, view, proj);
        
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
void handleGizmo(Mesh& mesh, const glm::mat4& view, const glm::mat4& proj) {
    static ImGuizmo::OPERATION op = ImGuizmo::TRANSLATE;
    
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetRect(0, 0, 
        ImGui::GetIO().DisplaySize.x, 
        ImGui::GetIO().DisplaySize.y);
    
    ImGuizmo::Manipulate(
        glm::value_ptr(view),
        glm::value_ptr(proj),
        op,
        ImGuizmo::WORLD,
        glm::value_ptr(mesh.transform)
    );
}