import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

###############################################################################
# 1. ADAPTIVE FUZZY MODEL (No Accuracy Tracking)
###############################################################################
class PneumoniaFuzzyModel:
    def __init__(self, window_size=10, alpha=0.3, cnn_confidence=0.248):
        self.membership_bounds = {
            "breathlessness": (0.0, 1.0),
            "sputum_production": (0.0, 1.0),
            "fever_duration": (0.0, 10.0),
            "fever_value": (35.0, 42.0),
            "hemoptysis": (0.0, 1.0),
            "fatigue": (0.0, 1.0),
            "appetite_loss": (0.0, 1.0),
            "confusion": (0.0, 1.0),
            "chest_pain": (0.0, 1.0),
            "cough_severity": (0.0, 1.0),
            "oxygen_level": (80.0, 100.0)
        }
        self.cnn_confidence = cnn_confidence # Placeholder for CNN confidence contribution

    def normalize(self, value, min_val, max_val):
        "Normalize value between 0 and 1"
        return (value - min_val) / (max_val - min_val)

    def fuzzify(self, symptom, value):
        """Computes fuzzy memberships using Mamdani-style membership functions."""
        # Normalize symptoms that need it
        if symptom == "fever_value":
            value = self.normalize(np.clip(value, 35.0, 42.0), 35.0, 42.0)
        elif symptom == "fever_duration":
            value = self.normalize(np.clip(value, 0.0, 10.0), 0.0, 10.0)
        elif symptom == "oxygen_level":
            value = self.normalize(np.clip(value, 80.0, 100.0), 80.0, 100.0)

        return {
            "poor": max(0, min(1, (0.3 - value) / 0.3)) if value <= 0.3 else 0,
            "average": max(0, min((value - 0.3) / 0.2, 1, (0.7 - value) / 0.2)) if 0.3 <= value <= 0.7 else 0,
            "good": max(0, min((value - 0.7) / 0.3, 1)) if value >= 0.7 else 0
        }

    def apply_rules(self, symptom_membership):
        """Applies fuzzy rules properly and integrates CNN confidence as a pivot factor."""
        severity_scores = {"severe": 0, "moderate": 0, "mild": 0, "negligible": 0}
        activated_rules = {"severe": [], "moderate": [], "mild": [], "negligible": []}

        for severity, rules in rules_by_severity.items():
            for rule_index, rule in enumerate(rules, start=1):
                logic_type = rule["logic_type"].upper()
                conditions = rule["conditions"]
                match_score = sum(1 for symptom, expected in conditions
                                  if symptom in symptom_membership and symptom_membership[symptom].get(expected, 0) > 0.5)
                condition_count = len(conditions)

                if logic_type == "AND" and match_score == condition_count:
                    severity_scores[severity] += 10
                    activated_rules[severity].append(f"Rule {rule_index} (AND)")
                elif logic_type == "OR" and match_score > 0:
                    severity_scores[severity] += 8
                    activated_rules[severity].append(f"Rule {rule_index} (OR)")
                elif logic_type == "XOR" and match_score == 1:
                    severity_scores[severity] += 9
                    activated_rules[severity].append(f"Rule {rule_index} (XOR)")
                elif logic_type == "NAND" and match_score < condition_count:
                    severity_scores[severity] += 7
                    activated_rules[severity].append(f"Rule {rule_index} (NAND)")

        # 📌 **CNN Confidence Pivot Application Before Choosing Severity**
        total_score = sum(severity_scores.values())
        if total_score > 0:
            cnn_weighting = {key: (value / total_score) * self.cnn_confidence for key, value in severity_scores.items()}
        else:
            cnn_weighting = {key: 0 for key in severity_scores.keys()}  # No impact if scores are zero

        # Apply CNN confidence weighting before final classification
        severity_scores = {key: severity_scores[key] + cnn_weighting[key] for key in severity_scores.keys()}

        # Final classification after CNN contribution as a pivot
        max_severity = max(severity_scores, key=severity_scores.get)

        return max_severity, activated_rules, severity_scores, cnn_weighting
    
###############################################################################
# 2. RULE DICTIONARY & HELPER FUNCTIONS
###############################################################################
rules_by_severity = {
    "severe": [
        # Rule 1: SEVERE if breathlessness is high, oxygen is low, confusion is high
        {
            "logic_type": "AND",
            "conditions": [
                ("breathlessness", "good"),
                ("oxygen_level", "poor"),
                ("confusion", "good")
            ]
        },
        # Rule 2: SEVERE if EITHER high fever-value OR prolonged fever-duration OR hemoptysis OR chest_pain
        {
            "logic_type": "OR",
            "conditions": [
                ("fever_value", "good"),
                ("fever_duration", "good"),
                ("hemoptysis", "good"),
                ("chest_pain", "good")
            ]
        },
        # Rule 3: SEVERE if EXACTLY ONE of (fatigue, appetite_loss) is "good" (XOR logic)
        {
            "logic_type": "XOR",
            "conditions": [
                ("fatigue", "good"),
                ("appetite_loss", "good")
            ]
        },
        # Rule 4: SEVERE if NOT (sputum_production=poor AND cough_severity=poor)
        {
            "logic_type": "NAND",
            "conditions": [
                ("sputum_production", "poor"),
                ("cough_severity", "poor")
            ]
        },
        # Rule 5: SEVERE if breathlessness and confusion are high, but fever is absent
        {
            "logic_type": "AND",
            "conditions": [
                ("breathlessness", "good"),
                ("confusion", "good"),
                ("fever_value", "poor")
            ]
        },
        # Rule 6: SEVERE if any of these symptoms are present: extreme fatigue, chest pain, or low oxygen
        {
            "logic_type": "OR",
            "conditions": [
                ("fatigue", "good"),
                ("chest_pain", "good"),
                ("oxygen_level", "poor")
            ]
        },
        # Rule 7: SEVERE if ONLY one of the following is severe: breathlessness, fever, confusion
        {
            "logic_type": "XOR",
            "conditions": [
                ("breathlessness", "good"),
                ("fever_value", "good"),
                ("confusion", "good")
            ]
        },
        # Rule 8: SEVERE if NOT (mild sputum production and mild cough severity)
        {
            "logic_type": "NAND",
            "conditions": [
                ("sputum_production", "average"),
                ("cough_severity", "average")
            ]
        }
    ],

    "moderate": [
        # Rule 1: MODERATE if breathlessness, oxygen_level, and fever_value are slightly high
        {
            "logic_type": "AND",
            "conditions": [
                ("breathlessness", "average"),
                ("oxygen_level", "average"),
                ("fever_value", "average")
            ]
        },
        # Rule 2: MODERATE if ANY of these symptoms are high: cough_severity, sputum_production, fatigue
        {
            "logic_type": "OR",
            "conditions": [
                ("cough_severity", "good"),
                ("sputum_production", "good"),
                ("fatigue", "good")
            ]
        },
        # Rule 3: MODERATE if hemoptysis OR chest pain are present but not severe (XOR)
        {
            "logic_type": "XOR",
            "conditions": [
                ("hemoptysis", "average"),
                ("chest_pain", "average")
            ]
        },
        # Rule 4: MODERATE if fever lasts long but is not extreme, or confusion is mild
        {
            "logic_type": "NOR",
            "conditions": [
                ("confusion", "average"),
                ("fever_duration", "average")
            ]
        },
        # Rule 5: MODERATE if fever is present but oxygen level is still normal
        {
            "logic_type": "AND",
            "conditions": [
                ("fever_value", "average"),
                ("oxygen_level", "good")
            ]
        },
        # Rule 6: MODERATE if ANY one of these symptoms is at moderate level: breathlessness, chest pain, fatigue
        {
            "logic_type": "OR",
            "conditions": [
                ("breathlessness", "average"),
                ("chest_pain", "average"),
                ("fatigue", "average")
            ]
        },
        # Rule 7: MODERATE if ONLY one of fever or confusion is present
        {
            "logic_type": "XOR",
            "conditions": [
                ("fever_value", "average"),
                ("confusion", "average")
            ]
        },
        # Rule 8: MODERATE if NOT (sputum production and cough severity both mild)
        {
            "logic_type": "NAND",
            "conditions": [
                ("sputum_production", "average"),
                ("cough_severity", "average")
            ]
        }
    ],

    "mild": [
         
         # Rule 1: MILD if NEITHER confusion nor prolonged fever
        {
            "logic_type": "NOR",
            "conditions": [
                ("confusion", "good"),
                ("fever_duration", "good")
            ]
        },
        # Rule 2: MILD if fever is present but duration is short
        {
            "logic_type": "AND",
            "conditions": [
                ("fever_value", "average"),
                ("fever_duration", "poor")
            ]
        },
        # Rule 3: MILD if ANY one of cough, sputum production, or appetite loss is present
        {
            "logic_type": "OR",
            "conditions": [
                ("cough_severity", "average"),
                ("sputum_production", "average"),
                ("appetite_loss", "average")
            ]
        },
        # Rule 4: MILD if ONLY fever OR fatigue is present
        {
            "logic_type": "XOR",
            "conditions": [
                ("fever_value", "average"),
                ("fatigue", "average")
            ]
        },
    ],

    "negligible": [
        # Rule 1: NEGLIGIBLE if oxygen level is normal AND fever is absent AND breathlessness is absent
        {
            "logic_type": "AND",
            "conditions": [
                ("oxygen_level", "good"),
                ("fever_value", "poor"),
                ("breathlessness", "poor")
            ]
        },
        # Rule 2: NEGLIGIBLE if either no productive cough OR no fever OR no chest pain
        {
            "logic_type": "OR",
            "conditions": [
                ("cough_severity", "poor"),
                ("fever_value", "poor"),
                ("chest_pain", "poor")
            ]
        },

        # Rule 3: NEGLIGIBLE if NOT (fever is high AND cough severity is high)
        {
            "logic_type": "NAND",
            "conditions": [
                ("fever_value", "good"),
                ("cough_severity", "good")
            ]
        },
        # Rule 4: NEGLIGIBLE if cough exists but no mucus/sputum production AND no fever
        {
            "logic_type": "AND",
            "conditions": [
                ("cough_severity", "average"),
                ("sputum_production", "poor"),
                ("fever_value", "poor")
            ]
        },
        # Rule 5: NEGLIGIBLE if ANY of the following are true: low fever, short fever duration, normal oxygen
        {
            "logic_type": "OR",
            "conditions": [
                ("fever_value", "average"),
                ("fever_duration", "poor"),
                ("oxygen_level", "good")
            ]
        },
        # Rule 6: NEGLIGIBLE if ONLY fever is present and no other symptoms
        {
            "logic_type": "XOR",
            "conditions": [
                ("fever_value", "average"),
                ("fatigue", "poor"),
                ("breathlessness", "poor"),
                ("cough_severity", "poor")
            ]
        },
        # Rule 7: NEGLIGIBLE if NOT (low oxygen and confusion present)
        {
            "logic_type": "NAND",
            "conditions": [
                ("oxygen_level", "good"),
                ("confusion", "good")
            ]
        }
    ]
}

###############################################################################
# 3. SYSTEMATIC TEST CASE GENERATION (Fixing Oxygen & Symptom Dependencies)
###############################################################################
def generate_realistic_symptoms(severity):
    """Generates realistic symptom values based strictly on the rules dictionary."""
    symptoms = {}

    # Fetch rules for the given severity
    relevant_rules = rules_by_severity[severity]

    for rule in relevant_rules:
        logic_type = rule["logic_type"].upper()
        conditions = rule["conditions"]

        for symptom, expected_membership in conditions:
            if expected_membership == "good":
                if symptom == "oxygen_level":
                    symptoms[symptom] = np.random.uniform(96.0, 100.0)  # ✅ Higher oxygen level for "good"
                elif symptom == "fever_value":
                    symptoms[symptom] = np.random.uniform(39.0, 42.0)  # ✅ High fever
                elif symptom == "fever_duration":
                    symptoms[symptom] = np.random.uniform(6.0, 10.0)  # ✅ Longer fever duration
                else:
                    symptoms[symptom] = np.random.uniform(0.7, 1.0)  # ✅ Other symptoms high

            elif expected_membership == "average":
                if symptom == "oxygen_level":
                    symptoms[symptom] = np.random.uniform(86.0, 95.0)  # ✅ Moderate oxygen
                elif symptom == "fever_value":
                    symptoms[symptom] = np.random.uniform(37.5, 38.9)
                elif symptom == "fever_duration":
                    symptoms[symptom] = np.random.uniform(3.1, 6.0)
                else:
                    symptoms[symptom] = np.random.uniform(0.3, 0.7)

            elif expected_membership == "poor":
                if symptom == "oxygen_level":
                    symptoms[symptom] = np.random.uniform(80.0, 85.0)  # ✅ Lower oxygen
                elif symptom == "fever_value":
                    symptoms[symptom] = np.random.uniform(35.0, 37.4)
                elif symptom == "fever_duration":
                    symptoms[symptom] = np.random.uniform(0.0, 3.0)
                else:
                    symptoms[symptom] = np.random.uniform(0.0, 0.3)

    # Ensure all symptoms are covered
    for symptom, (low, high) in PneumoniaFuzzyModel().membership_bounds.items():
        if symptom not in symptoms:
            symptoms[symptom] = np.random.uniform(low, high)

    return symptoms

###############################################################################
# 4. FUZZY TEST EXECUTION (WITH IMPROVED LOGIC)
###############################################################################
def run_fuzzy_tests(n_tests=10):
    model = PneumoniaFuzzyModel()
    severity_options = list(rules_by_severity.keys())

    for test_idx in range(1, n_tests + 1):
        chosen_severity = np.random.choice(severity_options)
        test_input = generate_realistic_symptoms(chosen_severity)
        symptom_membership = {s: model.fuzzify(s, v) for s, v in test_input.items()}

        # Compute severity before applying CNN confidence as a pivot
        actual_severity, activated_rules, severity_scores_before, cnn_weighting = model.apply_rules(symptom_membership)

        print(f"\n🩺 **Test Case {test_idx}:**")
        print(f"🔹 **Final Diagnosis (After CNN Pivot Influence):** {actual_severity.upper()}")
        print(f"🧠 **CNN Confidence Score Used as Pivot:** {model.cnn_confidence:.2f}")
        print("📌 **Passed Symptom Values:**")
        for symptom, value in test_input.items():
            print(f"  - {symptom}: {value:.2f}")

        print("\n📊 **Severity Scores Before CNN Pivot Contribution:**")
        for severity, score in severity_scores_before.items():
            print(f"  - {severity.capitalize()}: {score:.2f}")

        print("\n📊 **CNN Weighting Applied to Each Category Before Classification:**")
        for severity, weight in cnn_weighting.items():
            print(f"  - {severity.capitalize()}: {weight:.2f}")

        print(f"\n📊 **Activated Rules for {actual_severity.upper()}:**")
        if activated_rules[actual_severity]:
            for rule in activated_rules[actual_severity]:
                print(f"  ✅ {rule}")
        else:
            print("  ❌ No rules fired!")

# Run Tests
run_fuzzy_tests(10)


