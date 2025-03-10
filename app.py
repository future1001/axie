import requests
import cv2
import numpy as np
import base64
import logging
import time
import json
import tensorflow as tf
from typing import Tuple, Optional
from eth_account import Account
from eth_account.messages import encode_defunct
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from torchvision import transforms

# 配置日志（中文输出）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载验证码预测模型
MODEL_PATH = 'model_fold3.h5'
try:
    # 加载TensorFlow模型
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"TensorFlow模型加载成功！")
except Exception as e:
    logger.error(f"加载验证码模型失败: {e}")
    raise

def extract_features(image):
    """提取额外的CV特征"""
    # 将图像转换为OpenCV格式
    img = (image * 255).astype(np.uint8)
    
    # 转换为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 提取HOG特征（方向梯度直方图）- 非常适合角度检测
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    # 计算主方向（角度直方图的最高峰）
    hist, _ = np.histogram(angle, bins=12, range=(0, 360), weights=magnitude)
    dominant_angle = np.argmax(hist) * 30
    
    # 计算边缘强度在不同方向的分布
    edge_dirs = np.zeros(6)  # 6个方向(0°,60°,120°,180°,240°,300°)
    for i in range(6):
        # 计算该方向范围内的边缘强度总和
        dir_start = i * 60 - 30
        dir_end = i * 60 + 30
        if dir_start < 0:
            mask = (angle >= (dir_start + 360)) | (angle < dir_end)
        else:
            mask = (angle >= dir_start) & (angle < dir_end)
        edge_dirs[i] = np.sum(magnitude[mask])
    
    # 归一化
    if np.sum(edge_dirs) > 0:
        edge_dirs = edge_dirs / np.sum(edge_dirs)
    
    # 创建特征向量
    features = np.zeros(8)
    features[0] = dominant_angle / 360.0  # 归一化主方向
    features[1:7] = edge_dirs             # 各方向边缘强度
    features[7] = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else 0  # 方向一致性
    
    return features

def predict_angle(base64_image: str, max_retries: int = 3) -> int:
    """从Base64图片预测旋转角度，支持重试，适配TensorFlow模型"""
    for attempt in range(max_retries):
        try:
            # 解码Base64图像
            img_data = base64.b64decode(base64_image.split(',')[1])
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("无法解码图片")
            
            # 转换为RGB（OpenCV读取为BGR）
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 调整大小为224x224（与模型训练时一致）
            img = cv2.resize(img, (224, 224))
            
            # 标准化图像
            img_float = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_normalized = (img_float - mean) / std
            
            # 提取额外特征
            features = extract_features(img_float)
            
            # 准备模型输入
            img_batch = np.expand_dims(img_normalized, axis=0)
            features_batch = np.expand_dims(features, axis=0)
            
            # 预测
            predictions = MODEL.predict([img_batch, features_batch], verbose=0)
            angle_class = np.argmax(predictions[0])
            
            # 计算角度（类别*60度）
            angle = angle_class * 60
            logger.info(f"预测类别: {angle_class}, 预测角度: {angle}°")
            return int(angle)
        except Exception as e:
            logger.warning(f"第 {attempt + 1} 次预测角度失败: {e}")
            if attempt == max_retries - 1:
                logger.error(f"预测角度失败: {e}")
                raise
            time.sleep(3)

def load_proxies(file_path: str = "proxies.txt") -> list:
    """从proxies.txt加载代理列表"""
    try:
        with open(file_path, "r") as f:
            proxies = [line.strip() for line in f if line.strip()]
        if not proxies:
            raise ValueError("proxies.txt 文件为空")
        logger.info(f"加载了 {len(proxies)} 个代理")
        return proxies
    except Exception as e:
        logger.error(f"加载代理失败: {e}")
        raise

def get_captcha(session: requests.Session, check_url: str, app_key: str, proxies: list, proxy_index: int) -> Tuple[str, str]:
    """获取验证码图片和ID，超时或429时切换代理"""
    max_attempts = len(proxies)  # 最多尝试所有代理
    attempt = 0
    current_proxy_index = proxy_index

    while attempt < max_attempts:
        try:
            logger.info(f"使用代理 {proxies[current_proxy_index]} 获取验证码")
            payload = {"app_key": app_key}
            response = session.post(check_url, json=payload, timeout=5)  # 设置5秒超时
            response.raise_for_status()

            data = response.json()
            captcha_id = data.get("id")
            if not captcha_id:
                raise ValueError("响应中缺少验证码ID")

            base64_image = data.get("captcha_image") or data.get("image")
            if not base64_image:
                captcha_url = data.get("captcha_url")
                if captcha_url:
                    img_response = session.get(captcha_url, timeout=10)
                    img_response.raise_for_status()
                    img_data = img_response.content
                    base64_image = f"data:image/png;base64,{base64.b64encode(img_data).decode('utf-8')}"
                else:
                    raise ValueError("响应中缺少验证码图片数据")

            if not base64_image.startswith("data:image"):
                base64_image = f"data:image/png;base64,{base64_image}"

            if not base64_image.startswith("data:image"):
                raise ValueError(f"验证码图片格式错误: {base64_image[:50]}...")

            with open("captcha.png", "wb") as f:
                f.write(base64.b64decode(base64_image.split(',')[1]))
            logger.info(f"验证码图片已保存为 captcha.png，使用代理: {proxies[current_proxy_index]}")
            logger.info(f"获取到验证码ID: {captcha_id}")
            return captcha_id, base64_image

        except (requests.Timeout, requests.ConnectionError) as e:
            logger.warning(f"请求超时或连接错误: {e}，切换代理...")
            attempt += 1
            current_proxy_index = (current_proxy_index + 1) % len(proxies)
            session.close()
            session = configure_session(proxies[current_proxy_index])
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                logger.error(f"429 Too Many Requests，切换代理...")
                attempt += 1
                current_proxy_index = (current_proxy_index + 1) % len(proxies)
                session.close()
                session = configure_session(proxies[current_proxy_index])
            else:
                logger.error(f"HTTP错误: {e}")
                raise
        except Exception as e:
            logger.error(f"获取验证码失败: {e}")
            raise

    raise Exception("所有代理尝试失败，无法获取验证码")

def submit_captcha(session: requests.Session, submit_url: str, app_key: str, captcha_id: str, angle: int) -> str:
    """提交验证码结果并返回token"""
    logger.info("正在提交验证码结果...")
    payload = {"app_key": app_key, "id": captcha_id, "result": int(angle)}
    response = session.post(submit_url, json=payload, timeout=10)
    response.raise_for_status()
    result = response.json()
    logger.info(f"提交结果: {json.dumps(result, indent=2)}")
    token = result.get("token")
    if result.get("result", {}).get("status") != "right" or not token:
        raise ValueError(f"验证码提交失败: {json.dumps(result)}")
    logger.info(f"验证码 Token: {token[:50]}...")
    return token

def generate_signature(private_key: str, user_address: str) -> Tuple[str, int]:
    """生成签名，使用 {address}\nWelcome to Axie Infinity: Atia's Legacy!\n{timestamp} 格式"""
    try:
        timestamp = int(time.time())
        message = f"{user_address.lower()}\nWelcome to Axie Infinity: Atia's Legacy!\n{timestamp}"
        logger.info(f"完整消息: {message}")
        message_hash = encode_defunct(text=message)
        signed_message = Account.sign_message(message_hash, private_key=private_key)
        signature = "0x" + signed_message.signature.hex()
        logger.info(f"生成签名: {signature}")

        # 验证签名
        recovered_address = Account.recover_message(message_hash, signature=signature[2:])
        if recovered_address.lower() != user_address.lower():
            raise ValueError(f"签名验证失败！恢复地址 {recovered_address} 与预期地址 {user_address} 不匹配")
        logger.info(f"签名验证通过，恢复地址: {recovered_address}")
        return signature, timestamp
    except Exception as e:
        logger.error(f"签名生成失败: {e}")
        raise

def generate_wallet() -> Tuple[str, str]:
    """生成新的以太坊钱包地址和私钥"""
    try:
        account = Account.create()
        private_key = account.key.hex()
        address = account.address
        logger.info(f"生成新钱包 - 地址: {address}, 私钥: {private_key[:10]}...")
        return address, private_key
    except Exception as e:
        logger.error(f"钱包生成失败: {e}")
        raise

def register_wallet(session: requests.Session, wallet_index: int, check_url: str, submit_url: str, graphql_url: str, app_key: str, referred_by: str, proxies: list, proxy_index: int) -> Tuple[bool, str, str, str]:
    """注册单个钱包，返回是否成功及钱包信息，使用用户输入的邀请码"""
    # 添加代理轮询逻辑
    max_proxy_attempts = len(proxies)
    current_proxy_index = proxy_index
    
    for proxy_attempt in range(max_proxy_attempts):
        try:
            wallet_address, private_key = generate_wallet()
            logger.info(f"开始注册第 {wallet_index + 1} 个钱包, 地址: {wallet_address}")
            logger.info(f"使用代理 {proxies[current_proxy_index]} (尝试 {proxy_attempt + 1}/{max_proxy_attempts})")
            
            # 配置会话使用当前代理
            session.close()
            session = configure_session(proxies[current_proxy_index])
            
            captcha_id, base64_image = get_captcha(session, check_url, app_key, proxies, current_proxy_index)
            angle = predict_angle(base64_image)
            logger.info(f"预测的旋转角度: {angle}°")
            captcha_token = submit_captcha(session, submit_url, app_key, captcha_id, angle)
            
            signature, timestamp = generate_signature(private_key, wallet_address)
            
            # GraphQL请求部分
            graphql_query = {
                "query": """
                    mutation PreRegisterWithWallet(
                        $signature: String!
                        $referredBy: String
                        $timestamp: Int!
                        $userAddress: String!
                    ) {
                        atiaLegacyPreregisterWithWallet(
                            signature: $signature
                            timestamp: $timestamp
                            referredBy: $referredBy
                            userAddress: $userAddress
                        ) {
                            accessKey
                            registered
                        }
                    }
                """,
                "variables": {
                    "signature": signature,
                    "timestamp": timestamp,
                    "userAddress": wallet_address.lower(),
                    "referredBy": referred_by
                }
            }
            
            session.headers.update({"x-captcha": captcha_token})
            logger.info(f"GraphQL 请求 payload: {json.dumps(graphql_query, indent=2)}")
            logger.info(f"请求头: {session.headers}")
            logger.info("发送 GraphQL 请求...")
            
            response = session.post(graphql_url, json=graphql_query, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if "errors" in result:
                logger.warning(f"GraphQL 请求失败，错误详情: {json.dumps(result['errors'], indent=2)}")
                # 切换到下一个代理并重试
                current_proxy_index = (current_proxy_index + 1) % len(proxies)
                continue
            
            access_key = result["data"]["atiaLegacyPreregisterWithWallet"]["accessKey"]
            registered = result["data"]["atiaLegacyPreregisterWithWallet"]["registered"]
            logger.info(f"第 {wallet_index + 1} 个钱包注册结果: accessKey={access_key[:20]}..., registered={registered}")
            return registered, wallet_address, private_key, access_key
            
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"使用代理 {proxies[current_proxy_index]} 时出错: {e}")
            # 切换到下一个代理
            current_proxy_index = (current_proxy_index + 1) % len(proxies)
            
            # 如果已经尝试了所有代理，则返回失败
            if proxy_attempt == max_proxy_attempts - 1:
                logger.error(f"所有代理尝试失败，无法注册钱包")
                return False, wallet_address if 'wallet_address' in locals() else "", private_key if 'private_key' in locals() else "", ""
    
    # 如果所有尝试都失败
    return False, "", "", ""

def configure_session(proxy: str = None) -> requests.Session:
    """配置带重试的Session，支持代理"""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Content-Type": "application/json"
    })
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
        logger.info(f"设置代理: {proxy}")
    return session

def main():
    """主函数：自动生成并注册指定数量的钱包，每10个更换代理，使用用户输入的邀请码"""
    session = None
    try:
        logger.info("开始执行程序...")
        # 加载代理列表
        proxies = load_proxies("proxies.txt")
        proxy_index = 0
        session = configure_session(proxies[proxy_index])

        # 获取用户输入的邀请码
        referred_by = input("请输入您的邀请码（留空则不使用邀请码）：").strip() or None
        logger.info(f"使用邀请码: {referred_by if referred_by else '无'}")

        wallet_count = int(input("请输入需要注册的钱包数量：").strip())
        if wallet_count <= 0:
            raise ValueError("钱包数量必须大于0")
            
        # 设置代理轮询间隔
        proxy_rotation_interval = int(input("请输入每多少个钱包切换一次代理（建议5-10）：").strip() or "5")
        if proxy_rotation_interval <= 0:
            proxy_rotation_interval = 5
            logger.info(f"使用默认代理轮询间隔: {proxy_rotation_interval}")
        else:
            logger.info(f"设置代理轮询间隔: 每 {proxy_rotation_interval} 个钱包")

        app_key = "889a9cb7-3ffa-4113-9e3f-36558fe19808"
        check_url = "https://x.skymavis.com/captcha-srv/check"
        submit_url = "https://x.skymavis.com/captcha-srv/submit"
        graphql_url = "https://graphql-gateway.axieinfinity.com/graphql"
        
        successful_registrations = 0
        wallets = []
        
        for i in range(wallet_count):
            # 根据设置的间隔轮换代理
            if i > 0 and i % proxy_rotation_interval == 0:
                proxy_index = (proxy_index + 1) % len(proxies)  # 循环使用代理
                logger.info(f"更换代理，第 {i + 1} 个钱包使用新代理: {proxies[proxy_index]}")
                session.close()  # 关闭旧会话
                session = configure_session(proxies[proxy_index])  # 创建新会话并设置新代理

            registered, wallet_address, private_key, access_key = register_wallet(session, i, check_url, submit_url, graphql_url, app_key, referred_by, proxies, proxy_index)
            wallets.append({
                "address": wallet_address,
                "private_key": private_key,
                "registered": registered,
                "access_key": access_key,
                "referred_by": referred_by  # 添加邀请码到结果中
            })
            if registered:
                successful_registrations += 1
            time.sleep(5)  # 保持请求间隔
        
        with open("registered_wallets.json", "w") as f:
            json.dump(wallets, f, indent=2)
        logger.info(f"成功注册 {successful_registrations} 个钱包（共尝试 {wallet_count} 个），使用邀请码: {referred_by if referred_by else '无'}")
        logger.info("结果已保存到 registered_wallets.json")
        for idx, wallet in enumerate(wallets):
            logger.info(f"钱包 {idx + 1}: 地址={wallet['address']}, 私钥={wallet['private_key'][:20]}..., registered={wallet['registered']}, accessKey={wallet['access_key'][:20] if wallet['access_key'] else 'N/A'}, 邀请码={wallet['referred_by'] if wallet['referred_by'] else '无'}")
    except ValueError as e:
        logger.error(f"输入错误: {e}")
    except Exception as e:
        logger.error(f"主程序运行失败: {e}")
    finally:
        if session:
            session.close()
            logger.info("会话已关闭")

if __name__ == "__main__":
    main()