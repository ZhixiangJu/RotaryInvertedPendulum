#include <Wire.h>
#include <MPU6050.h>   // IMU 传感器（用于摆杆角度）
#include <Encoder.h>   // 增量编码器（用于旋转角度）

// ========== 1. 硬件引脚定义 ==========
#define MOTOR_PWM 9     // 电机 PWM 控制引脚
#define MOTOR_DIR 8     // 电机方向控制引脚
#define ENCODER_A 2     // 编码器 A 相（外部中断）
#define ENCODER_B 3     // 编码器 B 相

MPU6050 mpu;           // IMU 传感器对象
Encoder myEnc(ENCODER_A, ENCODER_B);  // 编码器对象

// ========== 2. 变量定义 ==========
float theta = 0.0, theta_dot = 0.0;  // 水平旋转角 & 角速度
float alpha = 0.0, alpha_dot = 0.0;  // 摆杆角 & 角速度
long last_encoder_pos = 0;
unsigned long last_time = 0;
const float ENCODER_RESOLUTION = 2048.0;  // 编码器分辨率

void setup() {
    Serial.begin(115200);  // 与 Jetson 进行串口通信
    Wire.begin();
    mpu.initialize();  // 初始化 IMU
    if (!mpu.testConnection()) {
        Serial.println("IMU 初始化失败");
        while (1);
    }

    pinMode(MOTOR_PWM, OUTPUT);
    pinMode(MOTOR_DIR, OUTPUT);
}

void loop() {
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');

        if (command.startsWith("SET_ACTION")) {
            int pwm_value = command.substring(11).toInt();
            applyPWM(pwm_value);
        } 
        else if (command.startsWith("GET_STATE")) {
            readSensors();
            Serial.print(theta); Serial.print(",");
            Serial.print(theta_dot); Serial.print(",");
            Serial.print(alpha); Serial.print(",");
            Serial.println(alpha_dot);
        } 
        else if (command.startsWith("RESET")) {
            applyPWM(0);
            while (!Serial.available());  // 等待手动扶正倒立摆
        }
    }
}

// ========== 传感器读取（θ, θ̇, α, α̇） ==========
void readSensors() {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    
    // 计算摆杆角度 α 和角速度 α̇
    alpha = atan2(ay, az) * 180.0 / PI;
    unsigned long current_time = millis();
    float dt = (current_time - last_time) / 1000.0;
    alpha_dot = (alpha - alpha_dot) / dt;
    last_time = current_time;
    
    // 计算旋转角 θ 和角速度 θ̇
    long encoder_pos = myEnc.read();
    theta = (encoder_pos / ENCODER_RESOLUTION) * 360.0;
    theta_dot = ((encoder_pos - last_encoder_pos) / ENCODER_RESOLUTION) * 360.0 / dt;
    last_encoder_pos = encoder_pos;
}

// ========== 控制电机 ==========
void applyPWM(int pwm) {
    pwm = constrain(pwm, -255, 255);
    analogWrite(MOTOR_PWM, abs(pwm));
    digitalWrite(MOTOR_DIR, pwm >= 0 ? HIGH : LOW);
}