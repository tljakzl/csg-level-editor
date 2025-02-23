#include "Camera.h"
#include <algorithm>

// Default camera values
const float SPEED = 15.0f;
const float SENSITIVTY = 0.25f;
const float ZOOM = 45.0f;

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch)
: Position(position)
, Front(glm::vec3(0.0f, 0.0f, -1.0f))
, Up(glm::vec3())
, Right(glm::vec3())
, WorldUp(up)
, Yaw(yaw)
, Pitch(pitch)
, MovementSpeed(SPEED)
, MouseSensitivity(SENSITIVTY)
, Zoom(ZOOM)
{
    UpdateCameraVectors();
}

Camera::Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch)
: Position(glm::vec3(posX, posY, posZ))
, Front(glm::vec3(0.0f, 0.0f, -1.0f))
, Up(glm::vec3())
, Right(glm::vec3())
, WorldUp(glm::vec3(upX, upY, upZ))
, Yaw(yaw)
, Pitch(pitch)
, MovementSpeed(SPEED)
, MouseSensitivity(SENSITIVTY)
, Zoom(ZOOM)
{
    UpdateCameraVectors();
}

void Camera::UpdateCameraVectors() {
    // Calculate the new Front vector
    glm::vec3 front;
    front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    front.y = sin(glm::radians(Pitch));
    front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    Front = glm::normalize(front);
    // Also re-calculate the Right and Up vector
    Right = glm::normalize(glm::cross(Front, WorldUp));  // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    Up = glm::normalize(glm::cross(Right, Front));
}

glm::mat4 Camera::GetViewMatrix(){
    return glm::lookAt(Position, Position + Front, Up);
}

void Camera::ProcessKeyboard(Camera::Movement direction, float deltaTime){
    GLfloat velocity = MovementSpeed * deltaTime;
    const glm::vec3 correction_dir(1.f, 1.f, 1.f);
    if (direction == FORWARD)
        Position += Front * correction_dir * velocity;
    if (direction == BACKWARD)
        Position -= Front * correction_dir * velocity;
    if (direction == LEFT)
        Position -= Right* correction_dir * velocity;
    if (direction == RIGHT)
        Position += Right* correction_dir * velocity;
}

void Camera::Update(float dt)
{
}

void Camera::ProcessMouseMovement(float xoffset, float yoffset, GLboolean constrainPitch){
    xoffset *= MouseSensitivity;
    yoffset *= MouseSensitivity;

    Yaw += xoffset;
    Pitch += yoffset;

    // Make sure that when pitch is out of bounds, screen doesn't get flipped
    if (constrainPitch)
    {
        if (Pitch > 89.0f)
            Pitch = 89.0f;
        if (Pitch < -89.0f)
            Pitch = -89.0f;
    }

    // Update Front, Right and Up Vectors using the updated Eular angles
    UpdateCameraVectors();
}

void Camera::ProcessMouseScroll(float yoffset)	{
    if (Zoom >= 1.0f && Zoom <= 45.0f)
        Zoom -= yoffset;
    if (Zoom <= 1.0f)
        Zoom = 1.0f;
    if (Zoom >= 45.0f)
        Zoom = 45.0f;
}

const glm::vec3 &Camera::GetPosition() const {
    return Position;
}

const glm::vec3& Camera::GetFront() const
{
    return Front;
}

float Camera::GetZoom() const {
    return Zoom;
}
