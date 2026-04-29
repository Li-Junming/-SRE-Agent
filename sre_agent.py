import os
import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# ==========================================
# 1. 定义运维操作工具 (Tools)
# ==========================================

@tool
def get_service_metrics(cloud_provider: str, service_name: str) -> str:
    """
    获取指定云平台（AWS, Azure, GCP）中具体服务的实时运行指标（CPU、内存、延迟）。
    """
    # 生产环境中，这里会替换为调用 AWS CloudWatch 或 Azure Monitor API
    mock_data = {
        "aws": {
            "order-service": {"cpu": "98%", "memory": "85%", "latency": "5.2s", "status": "Critical"},
            "payment-service": {"cpu": "45%", "memory": "60%", "latency": "0.3s", "status": "Healthy"}
        },
        "azure": {
            "user-db": {"connections": "95%", "iops": "99%", "status": "Warning"}
        }
    }
    
    provider_data = mock_data.get(cloud_provider.lower(), {})
    service_data = provider_data.get(service_name.lower(), {"error": "Service not found"})
    return json.dumps({f"{cloud_provider}-{service_name}": service_data})

@tool
def query_application_logs(service_name: str, time_range: str = "last_15_mins") -> str:
    """
    查询指定服务的最近日志，用于排查错误堆栈或异常关键字。
    """
    # 生产环境中，这里会替换为调用 Datadog, ELK 或 AWS CloudWatch Logs API
    mock_logs = {
        "order-service": [
            "[ERROR] Connection timed out while calling inventory-service.",
            "[FATAL] OutOfMemoryError: Java heap space. Process exiting."
        ],
        "user-db": [
            "[WARN] Too many connections. Reached max_connections limit."
        ]
    }
    
    logs = mock_logs.get(service_name.lower(), ["No recent logs found."])
    return "\n".join(logs)

@tool
def execute_remediation(cloud_provider: str, action: str, target: str) -> str:
    """
    执行故障恢复操作（如 restart_service, scale_up_instances）。
    注意：这是高风险操作。
    """
    # 生产环境中，这里会替换为调用 Kubernetes API 或云服务 SDK
    allowed_actions = ["restart_service", "scale_up_instances"]
    if action not in allowed_actions:
        return f"执行失败：不支持的操作 '{action}'。"
    
    return f"✅ 【操作成功】已在 {cloud_provider.upper()} 平台对目标 {target} 执行了 {action} 操作。"

# ==========================================
# 2. 封装智能 SRE Agent
# ==========================================

class MultiCloudSREAgent:
    def __init__(self, model_name="gpt-4o"):
        # 初始化大模型 (建议使用推理能力强的模型)
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # 注册工具集
        self.tools = [get_service_metrics, query_application_logs, execute_remediation]
        
        # 设定 SRE 专家的系统提示词 (System Prompt)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个资深的智能多云 SRE (运维) 专家。
            你的主要任务是处理多云环境（如 AWS, Azure）下的生产故障告警。
            
            处理原则：
            1. 验证：收到告警后，必须先使用指标工具验证系统当前状态。
            2. 诊断：结合系统指标和应用日志进行根因分析（RCA）。
            3. 修复：如果确定了根因且有现成的修复手段（如重启、扩容），请执行修复。
            4. 严谨：在执行修复前，明确说明你的判断逻辑。如果信息不足以进行自动修复，请给出详细的手动排查建议，切勿盲目操作。
            """),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 构建 Agent 执行器
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def handle_incident(self, alert_message: str) -> str:
        print(f"\n🚨 收到新告警: {alert_message}")
        print("-" * 50)
        
        # 触发 Agent 运行
        response = self.executor.invoke({"input": alert_message})
        return response["output"]

# ==========================================
# 3. 运行测试
# ==========================================

if __name__ == "__main__":
    # 检查 API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("错误：请先设置 OPENAI_API_KEY 环境变量。")
        exit(1)
        
    sre_agent = MultiCloudSREAgent()

    # 模拟一个涉及 AWS 的复杂生产故障
    incident_1 = "P1 告警：AWS 上的 order-service 响应时间飙升并出现大量 502 错误。请立刻排查并恢复服务。"
    report_1 = sre_agent.handle_incident(incident_1)
    
    print("\n📝 SRE Agent 最终处理报告:")
    print(report_1)
    
    print("\n" + "="*50)
    
    # 模拟一个涉及 Azure 的预警
    incident_2 = "预警通知：Azure 区域的 user-db 告警，请检查数据库状况并给出处理建议。"
    report_2 = sre_agent.handle_incident(incident_2)
    
    print("\n📝 SRE Agent 最终处理报告:")
    print(report_2)