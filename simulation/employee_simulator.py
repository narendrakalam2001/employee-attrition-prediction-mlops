# ============================================================
# EMPLOYEE SIMULATOR — Employee Attrition ML System
# ============================================================

import requests
import random
import time
import os

# ── API URL — local dev vs deployed ──────────────────────────
API_URL = os.getenv(
    "ATTRITION_API_URL", "http://127.0.0.1:8000"
) + "/predict"


# ============================================================
# GENERATE SYNTHETIC EMPLOYEE
# ============================================================

def generate_employee(scenario: str = "random") -> dict:
    """
    Scenarios:
        random    — mixed realistic workforce
        at_risk   — high attrition-risk profile (overworked, dissatisfied, stagnant)
        loyal     — low attrition-risk profile (settled, satisfied, growing)
    """

    if scenario == "at_risk":
        return {
            "age":                      random.randint(22, 32),
            "businesstravel":           random.choice(["Travel_Frequently", "Travel_Rarely"]),
            "dailyrate":                random.uniform(200, 600),
            "department":               random.choice(["Sales", "Research & Development"]),
            "distancefromhome":         random.randint(20, 30),
            "education":                random.choice([1, 2]),
            "educationfield":           random.choice(["Life Sciences", "Marketing"]),
            "environmentsatisfaction":  random.choice([1, 2]),
            "gender":                   random.choice(["Male", "Female"]),
            "hourlyrate":               random.uniform(30, 60),
            "jobinvolvement":           random.choice([1, 2]),
            "joblevel":                 1,
            "jobrole":                  random.choice(["Sales Representative", "Laboratory Technician"]),
            "jobsatisfaction":          random.choice([1, 2]),
            "maritalstatus":            "Single",
            "monthlyincome":            random.uniform(15000, 30000),
            "monthlyrate":              random.uniform(10000, 20000),
            "numcompaniesworked":       random.randint(3, 8),
            "overtime":                 "Yes",
            "percentsalaryhike":        random.randint(11, 14),
            "performancerating":        3,
            "relationshipsatisfaction": random.choice([1, 2]),
            "stockoptionlevel":         0,
            "totalworkingyears":        random.randint(1, 5),
            "trainingtimeslastyear":    random.randint(0, 1),
            "worklifebalance":          random.choice([1, 2]),
            "yearsatcompany":           random.randint(0, 2),
            "yearsincurrentrole":       random.randint(0, 1),
            "yearssincelastpromotion":  random.randint(0, 1),
            "yearswithcurrmanager":     random.randint(0, 1),
        }

    elif scenario == "loyal":
        return {
            "age":                      random.randint(38, 58),
            "businesstravel":           "Travel_Rarely",
            "dailyrate":                random.uniform(700, 1400),
            "department":               random.choice(["Research & Development", "Human Resources"]),
            "distancefromhome":         random.randint(1, 8),
            "education":                random.choice([3, 4, 5]),
            "educationfield":           random.choice(["Life Sciences", "Medical"]),
            "environmentsatisfaction":  random.choice([3, 4]),
            "gender":                   random.choice(["Male", "Female"]),
            "hourlyrate":               random.uniform(70, 100),
            "jobinvolvement":           random.choice([3, 4]),
            "joblevel":                 random.choice([3, 4, 5]),
            "jobrole":                  random.choice(["Manager", "Research Director", "Healthcare Representative"]),
            "jobsatisfaction":          random.choice([3, 4]),
            "maritalstatus":            "Married",
            "monthlyincome":            random.uniform(90000, 180000),
            "monthlyrate":              random.uniform(15000, 26000),
            "numcompaniesworked":       random.randint(0, 2),
            "overtime":                 "No",
            "percentsalaryhike":        random.randint(14, 22),
            "performancerating":        random.choice([3, 4]),
            "relationshipsatisfaction": random.choice([3, 4]),
            "stockoptionlevel":         random.choice([1, 2, 3]),
            "totalworkingyears":        random.randint(15, 30),
            "trainingtimeslastyear":    random.randint(2, 5),
            "worklifebalance":          random.choice([3, 4]),
            "yearsatcompany":           random.randint(10, 25),
            "yearsincurrentrole":       random.randint(3, 10),
            "yearssincelastpromotion":  random.randint(0, 3),
            "yearswithcurrmanager":     random.randint(3, 10),
        }

    else:  # random
        return {
            "age":                      random.randint(22, 60),
            "businesstravel":           random.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"]),
            "dailyrate":                random.uniform(100, 1500),
            "department":               random.choice(["Sales", "Research & Development", "Human Resources"]),
            "distancefromhome":         random.randint(1, 30),
            "education":                random.randint(1, 5),
            "educationfield":           random.choice(["Life Sciences", "Medical", "Marketing", "Technical Degree"]),
            "environmentsatisfaction":  random.randint(1, 4),
            "gender":                   random.choice(["Male", "Female"]),
            "hourlyrate":               random.uniform(30, 100),
            "jobinvolvement":           random.randint(1, 4),
            "joblevel":                 random.randint(1, 5),
            "jobrole":                  random.choice(["Sales Executive", "Research Scientist", "Manager", "Manufacturing Director"]),
            "jobsatisfaction":          random.randint(1, 4),
            "maritalstatus":            random.choice(["Single", "Married", "Divorced"]),
            "monthlyincome":            random.uniform(15000, 200000),
            "monthlyrate":              random.uniform(10000, 27000),
            "numcompaniesworked":       random.randint(0, 9),
            "overtime":                 random.choice(["Yes", "No"]),
            "percentsalaryhike":        random.randint(11, 25),
            "performancerating":        random.choice([3, 4]),
            "relationshipsatisfaction": random.randint(1, 4),
            "stockoptionlevel":         random.randint(0, 3),
            "totalworkingyears":        random.randint(0, 40),
            "trainingtimeslastyear":    random.randint(0, 6),
            "worklifebalance":          random.randint(1, 4),
            "yearsatcompany":           random.randint(0, 40),
            "yearsincurrentrole":       random.randint(0, 18),
            "yearssincelastpromotion":  random.randint(0, 15),
            "yearswithcurrmanager":     random.randint(0, 17),
        }


# ============================================================
# SEND TO API + PRINT RESULT
# ============================================================

def send_employee(employee: dict, idx: int):

    try:
        response = requests.post(API_URL, json=employee, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print(f"[{idx+1}]  Income={employee['monthlyincome']:.0f}  "
                  f"Tenure={employee['yearsatcompany']}yr  "
                  f"OT={employee['overtime']}  "
                  f"→  prob={result['attrition_probability']:.4f}  "
                  f"band={result['risk_band']}  "
                  f"decision={result['decision']}")
        else:
            print(f"[{idx+1}] API error: {response.status_code}")

    except Exception as e:
        print(f"[{idx+1}] Connection error: {e}")


# ============================================================
# RUN SIMULATION
# ============================================================

def simulate_workforce(n: int = 20, scenario: str = "random"):

    print(f"\nSimulating {n} employees  |  scenario={scenario}\n" + "-" * 60)

    for i in range(n):
        employee = generate_employee(scenario)
        send_employee(employee, i)
        time.sleep(0.5)

    print("-" * 60 + "\nSimulation complete")


if __name__ == "__main__":
    simulate_workforce(20, scenario="random")
