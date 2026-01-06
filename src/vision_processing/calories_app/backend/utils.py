def calculate_nutriscore(kcal, sugar, sat_fat, sodium, fruit_pct, fiber, protein):
    """
    Approximation simplifiée du Nutri-Score (FSAm-NPS score).
    Points N (Négatifs): Calories, Sucre, Acides gras saturés, Sodium
    Points P (Positifs): Fruits/Légumes, Fibres, Protéines
    Score = Total N - Total P
    """
    # 1. Points N (0-40)
    points_n = 0
    # Energie (kJ)
    kj = kcal * 4.184
    if kj > 3350: points_n += 10
    elif kj > 0: points_n += int(kj / 335)
    
    # Sucre (g)
    if sugar > 45: points_n += 10
    elif sugar > 0: points_n += int(sugar / 4.5)
    
    # Gras Saturé (g) - Approx: on assume 30% du gras total si inconnu
    if sat_fat > 10: points_n += 10
    elif sat_fat > 0: points_n += int(sat_fat / 1)
    
    # Sodium (mg) - Approx: on assume 500mg par défaut si inconnu
    points_n += 2 # Valeur moyenne
    
    # 2. Points P (0-15)
    points_p = 0
    # Fibres (g)
    if fiber > 4.7: points_p += 5
    elif fiber > 0: points_p += int(fiber / 0.9)
    
    # Protéines (g)
    if protein > 8: points_p += 5
    elif protein > 0: points_p += int(protein / 1.6)
    
    score_final = points_n - points_p
    
    # Conversion en Classe (A-E)
    # A: <= -1
    # B: 0 - 2
    # C: 3 - 10
    # D: 11 - 18
    # E: >= 19
    if score_final <= -1: return "A", 100 - (score_final + 15) * 2 # Score fictif 80-100
    if score_final <= 2: return "B", 80 - score_final * 5
    if score_final <= 10: return "C", 60 - (score_final-2) * 2
    if score_final <= 18: return "D", 40 - (score_final-10) * 2
    return "E", 20
