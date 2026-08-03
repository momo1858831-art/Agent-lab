from typing import List,Dict,Any,Optional

# 存储智能体的行动与反思
class Memory:

    def __init__(self):
        self.records:List[Dict[str,Any]]=[]

    # 添加新纪录
    # 反思 or 行动 + 具体内容
    def add_record(self,record_type:str,content:str):
        record={"type":record_type,"content":content}
        self.records.append(record)
        print(f"\n新增一条{record_type}记忆")

    # 获取所有记录
    def get_trajectory(self):
        trajectory_parts=[]
        for record in self.records:
            if record['type']=='execution':
                trajectory_parts.append(f"上一轮执行结果为:\n{record['content']}")
            elif record['type']=='reflection':
                trajectory_parts.append(f"反思结果为:\n{record['content']}")
        return "\n\n".join(trajectory_parts)

    # 获取最近一次执行结果
    def get_last_execution(self):
        for record in reversed(self.records):
            if record['type']=='execution':
                return record['content']
        return None

    