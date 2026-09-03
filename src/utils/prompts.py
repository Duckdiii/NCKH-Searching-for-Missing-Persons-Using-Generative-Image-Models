"""
Hàm tiện ích dùng chung cho cả 3 module của FADING (Specialization, Inversion, Editing):
suy tuổi đại diện từ age_group, quy đổi gender sang từ mô tả, và build các prompt
P_alpha / P_neutral / P_tau theo đúng Enhanced Prompt (EP) của paper.
"""

from typing import Dict

# Trung điểm từng age_group trong sampled_labels.csv.
# Riêng nhóm cuối "70-120" LẤY TAY = 80, không dùng trung điểm toán học (sẽ ra 95),
# vì paper FADING gốc ghi rõ: "For the oldest age group (70+), we translate to 80 years old".
AGE_GROUP_TO_AGE: Dict[str, int] = {
    "0-2": 1,
    "3-6": 4,
    "7-9": 8,
    "10-14": 12,
    "15-19": 17,
    "20-29": 24,
    "30-39": 34,
    "40-49": 44,
    "50-69": 59,
    "70-120": 80,
}


def age_group_to_age(age_group: str) -> int:
    """Quy đổi 1 nhãn age_group (vd "30-39") sang 1 con số tuổi đại diện (vd 34)."""
    return AGE_GROUP_TO_AGE[age_group]


def gender_to_word(gender: str, age: int) -> str:
    """Quy đổi gender ("male"/"female") + tuổi sang từ mô tả giới tính dùng trong prompt:
    woman/man cho người lớn (age >= 15), girl/boy nếu age < 15."""
    is_female = gender.lower() == "female"
    if age < 15:
        return "girl" if is_female else "boy"
    return "woman" if is_female else "man"


def build_prompt_alpha(age: int, gender_word: str) -> str:
    """Build P_alpha = "photo of a {age} year old {gender_word}" - prompt có tuổi,
    dùng trong Module 1 (nhánh alpha) và Module 2 (Initial Age)."""
    return f"photo of a {age} year old {gender_word}"


def build_prompt_neutral(gender_word: str) -> str:
    """Build P_neutral = "photo of a {gender_word}" - prompt trung lập, không chứa tuổi,
    dùng trong Module 1 (nhánh neutral)."""
    return f"photo of a {gender_word}"


def build_prompt_tau(target_age: int, gender_word: str) -> str:
    """Build P_tau = "photo of a {target_age} year old {gender_word}" - prompt cho target_age,
    dùng trong Module 3 (Editing). Cùng công thức với P_alpha, tách hàm riêng cho rõ ngữ nghĩa
    sử dụng (P_alpha ứng với Initial Age, P_tau ứng với target age cần sinh ảnh)."""
    return build_prompt_alpha(target_age, gender_word)
