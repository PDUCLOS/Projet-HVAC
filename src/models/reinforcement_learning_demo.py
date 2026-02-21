# -*- coding: utf-8 -*-
"""
Apprentissage par Renforcement — Optimisation de la maintenance HVAC (Phase 5).
================================================================================

Demonstration pedagogique du Reinforcement Learning (RL) applique a
l'optimisation de la planification de maintenance HVAC sur 96 departements
francais.

Concepts RL :
    - **Agent** : Systeme de decision qui observe l'etat et choisit une action.
    - **Environnement** : Simulateur HVAC retournant etat + recompense.
    - **Etat** : Vecteur [temperature, humidite, age, demande, saison, budget].
    - **Action** : {ne rien faire, preventive, corrective, remplacement}.
    - **Recompense** : Signal de qualite (+preventif avant panne, -panne, -cout).
    - **Politique** : Mapping etat -> action maximisant la recompense cumulee.

Algorithme Q-Learning :
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
    Algorithme model-free, off-policy. Table Q discretisee, exploration
    epsilon-greedy avec decroissance exponentielle.

Passage a l'echelle :
    - Stable-Baselines3 (PPO/SAC) pour du Deep RL avec PyTorch
    - Ray RLlib pour l'entrainement distribue multi-agent

Dependances : numpy, matplotlib, gymnasium (optionnel)

Usage :
    python src/models/reinforcement_learning_demo.py
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- Gestion optionnelle de Gymnasium ---
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    warnings.warn(
        "gymnasium n'est pas installe. Interface simplifiee utilisee. "
        "Installez-le via : pip install gymnasium"
    )

# --- Matplotlib (mode non-interactif pour serveurs sans affichage) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Environnement personnalise — HVACMaintenanceEnv
# =============================================================================

# Constantes du domaine HVAC
ACTIONS = {0: "aucune_maintenance", 1: "maintenance_preventive",
           2: "reparation_corrective", 3: "remplacement_equipement"}
COUTS_ACTIONS = {0: 0.0, 1: 0.15, 2: 0.40, 3: 0.70}
MOIS = ["Janvier", "Fevrier", "Mars", "Avril", "Mai", "Juin",
        "Juillet", "Aout", "Septembre", "Octobre", "Novembre", "Decembre"]


class HVACMaintenanceEnv:
    """Environnement Gymnasium simulant la maintenance HVAC sur 12 mois.

    Observation (Box, 6 dim) : [temp, humidite, age, demande, saison, budget]
    Actions (Discrete 4) : aucune / preventive / corrective / remplacement
    Episode : 12 pas (1 an), recompense selon qualite des decisions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None):
        """Initialise l'environnement HVAC."""
        super().__init__()
        self.render_mode = render_mode
        self.observation_space_low = np.array([0.0] * 6, dtype=np.float32)
        self.observation_space_high = np.array([1.0] * 6, dtype=np.float32)
        self.n_actions = 4

        if GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=self.observation_space_low,
                high=self.observation_space_high, dtype=np.float32)
            self.action_space = spaces.Discrete(self.n_actions)

        # Variables internes
        self.mois_courant: int = 0
        self.budget_restant: float = 1.0
        self.age_equipement: float = 0.0
        self.etat_equipement: float = 1.0  # 1.0 = neuf, 0.0 = hors service
        self.np_random = np.random.default_rng()

    def _generer_meteo(self, mois: int) -> Tuple[float, float]:
        """Genere temperature et humidite realistes (climat francais tempere)."""
        # Temperature sinusoidale : ~3 C en janvier, ~22 C en juillet
        temp = 12.5 + 10.0 * np.sin(2 * np.pi * (mois - 3) / 12)
        temp += self.np_random.normal(0, 2.0)
        temperature = float(np.clip((temp + 10) / 50, 0.0, 1.0))
        # Humidite : plus elevee en hiver
        hum = 0.7 - 0.15 * np.sin(2 * np.pi * (mois - 3) / 12)
        humidite = float(np.clip(hum + self.np_random.normal(0, 0.05), 0.0, 1.0))
        return temperature, humidite

    def _calculer_demande(self, temperature: float, mois: int) -> float:
        """Demande HVAC en U : forte en hiver (chauffage) et ete (clim)."""
        temp_reelle = temperature * 50 - 10
        demande = abs(temp_reelle - 20) / 30 + self.np_random.normal(0, 0.05)
        return float(np.clip(demande, 0.0, 1.0))

    def _probabilite_panne(self) -> float:
        """Probabilite de panne croissante avec l'age, decroissante avec l'etat."""
        prob = 0.05 + 0.4 * (self.age_equipement ** 2) * (1 - self.etat_equipement * 0.5)
        return float(np.clip(prob, 0.0, 0.95))

    def _construire_observation(self, temp: float, hum: float, dem: float) -> np.ndarray:
        """Construit le vecteur d'observation [temp, hum, age, dem, saison, budget]."""
        saison = (np.sin(2 * np.pi * self.mois_courant / 12) + 1) / 2
        return np.array([temp, hum, self.age_equipement, dem,
                         saison, self.budget_restant], dtype=np.float32)

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict]:
        """Reinitialise l'episode avec conditions initiales aleatoires."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.mois_courant = 0
        self.budget_restant = 1.0
        self.age_equipement = float(self.np_random.uniform(0.0, 0.6))
        self.etat_equipement = float(np.clip(
            1.0 - self.age_equipement * 0.5 + self.np_random.normal(0, 0.1), 0.2, 1.0))

        temp, hum = self._generer_meteo(0)
        obs = self._construire_observation(temp, hum, self._calculer_demande(temp, 0))
        return obs, {"mois": MOIS[0], "age_initial": self.age_equipement}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute une action et avance d'un mois. Retourne (obs, rew, done, trunc, info)."""
        assert 0 <= action < self.n_actions, f"Action invalide : {action}"
        recompense = 0.0
        info: Dict[str, Any] = {"mois": MOIS[self.mois_courant], "action": ACTIONS[action]}

        # 1. Cout de l'action
        cout = COUTS_ACTIONS[action]
        self.budget_restant = max(0.0, self.budget_restant - cout)
        recompense -= cout

        # 2. Effet de l'action + pannes
        panne_survenue = False
        prob_panne = self._probabilite_panne()
        info["probabilite_panne"] = round(prob_panne, 3)

        if action == 0:
            # Aucune maintenance — risque de panne
            if self.np_random.random() < prob_panne:
                panne_survenue = True
                recompense -= 5.0
                self.etat_equipement = max(0.1, self.etat_equipement - 0.4)
        elif action == 1:
            # Maintenance preventive
            if self.age_equipement > 0.3:
                recompense += 2.0  # Bonne decision sur equipement vieillissant
                self.etat_equipement = min(1.0, self.etat_equipement + 0.2)
                if self.np_random.random() < prob_panne:
                    recompense += 3.0  # Bonus : panne evitee
            else:
                recompense -= 0.5  # Maintenance inutile, equipement neuf
        elif action == 2:
            # Reparation corrective
            if self.etat_equipement < 0.6:
                self.etat_equipement = min(1.0, self.etat_equipement + 0.35)
                recompense += 1.0
            else:
                recompense -= 0.3  # Reparation inutile
        elif action == 3:
            # Remplacement complet
            if self.age_equipement > 0.7:
                self.age_equipement, self.etat_equipement = 0.0, 1.0
                recompense += 2.5
            else:
                self.age_equipement, self.etat_equipement = 0.0, 1.0
                recompense -= 1.0  # Remplacement premature

        # 3. Vieillissement naturel
        self.age_equipement = min(1.0, self.age_equipement + 1.0 / 15)
        self.etat_equipement = max(0.1, self.etat_equipement - 0.03)

        # 4. Avancement temporel
        self.mois_courant += 1
        termine = self.mois_courant >= 12

        # Bonus fin d'episode : bonne gestion budgetaire
        if termine:
            recompense += self.budget_restant * 2.0

        # 5. Nouvel etat
        mois_obs = min(self.mois_courant, 11)
        temp, hum = self._generer_meteo(mois_obs)
        obs = self._construire_observation(temp, hum, self._calculer_demande(temp, mois_obs))
        info.update({"panne": panne_survenue, "etat_equipement": round(self.etat_equipement, 3),
                     "budget_restant": round(self.budget_restant, 3)})
        return obs, float(recompense), termine, False, info


# =============================================================================
# 2. Agent Q-Learning (implementation de zero, sans librairie RL externe)
# =============================================================================

class AgentQLearning:
    """Agent Q-Learning tabular avec discretisation d'etat et epsilon-greedy.

    Exploration vs exploitation : epsilon commence a 1.0 (exploration totale)
    et decroit exponentiellement vers epsilon_min (exploitation dominante).
    """

    def __init__(self, n_bins: int = 6, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995):
        """Configure les hyperparametres du Q-Learning."""
        self.n_bins = n_bins
        self.alpha = alpha      # Taux d'apprentissage
        self.gamma = gamma      # Facteur d'actualisation
        self.epsilon = epsilon   # Probabilite d'exploration
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = 4
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}

    def _discretiser_etat(self, observation: np.ndarray) -> Tuple[int, ...]:
        """Discretise l'observation continue en n_bins par dimension."""
        indices = np.clip((observation * self.n_bins).astype(int), 0, self.n_bins - 1)
        return tuple(indices.tolist())

    def _obtenir_valeurs_q(self, etat: Tuple[int, ...]) -> np.ndarray:
        """Retourne les valeurs Q pour un etat (initialise a 0 si inconnu)."""
        if etat not in self.q_table:
            self.q_table[etat] = np.zeros(self.n_actions, dtype=np.float64)
        return self.q_table[etat]

    def obtenir_q_plus_proche(self, observation: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Retourne les Q du bin exact ou du voisin le plus proche si inexploite.

        Utile pour l'affichage de la politique sur des etats non visites.
        Retourne (q_values, exact_match).
        """
        etat = self._discretiser_etat(observation)
        if etat in self.q_table:
            return self.q_table[etat].copy(), True
        # Recherche du voisin le plus proche (distance de Manhattan)
        if not self.q_table:
            return np.zeros(self.n_actions), False
        etat_arr = np.array(etat)
        meilleur, dist_min = None, float("inf")
        for e_visite in self.q_table:
            d = np.sum(np.abs(np.array(e_visite) - etat_arr))
            if d < dist_min:
                dist_min, meilleur = d, e_visite
        return self.q_table[meilleur].copy(), False

    def choisir_action(self, observation: np.ndarray) -> int:
        """Epsilon-greedy : exploration aleatoire ou exploitation de Q max."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        etat = self._discretiser_etat(observation)
        return int(np.argmax(self._obtenir_valeurs_q(etat)))

    def mettre_a_jour(self, obs: np.ndarray, action: int, recompense: float,
                      obs_suiv: np.ndarray, termine: bool) -> None:
        """Mise a jour Q-Learning : Q(s,a) += alpha * [cible - Q(s,a)]."""
        etat = self._discretiser_etat(obs)
        q_vals = self._obtenir_valeurs_q(etat)
        q_suiv = self._obtenir_valeurs_q(self._discretiser_etat(obs_suiv))
        cible = recompense if termine else recompense + self.gamma * np.max(q_suiv)
        q_vals[action] += self.alpha * (cible - q_vals[action])

    def decroitre_epsilon(self) -> None:
        """Decroissance exponentielle d'epsilon apres chaque episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# =============================================================================
# 3. Boucles d'entrainement et comparaison avec politique aleatoire
# =============================================================================

def entrainer_agent(env: HVACMaintenanceEnv, agent: AgentQLearning,
                    n_episodes: int = 1000) -> List[float]:
    """Entraine l'agent Q-Learning. Retourne l'historique des recompenses."""
    historique: List[float] = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        cumul = 0.0
        for _ in range(12):
            action = agent.choisir_action(obs)
            obs_suiv, rew, done, trunc, _ = env.step(action)
            agent.mettre_a_jour(obs, action, rew, obs_suiv, done)
            cumul += rew
            obs = obs_suiv
            if done or trunc:
                break
        agent.decroitre_epsilon()
        historique.append(cumul)
        if (episode + 1) % 200 == 0:
            logger.info(f"Episode {episode+1}/{n_episodes} — "
                        f"Moy(100) : {np.mean(historique[-100:]):.2f} — "
                        f"Eps : {agent.epsilon:.3f}")
    return historique


def evaluer_politique_aleatoire(env: HVACMaintenanceEnv,
                                n_episodes: int = 1000) -> List[float]:
    """Baseline : politique purement aleatoire pour comparaison."""
    historique: List[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        cumul = 0.0
        for _ in range(12):
            obs, rew, done, trunc, _ = env.step(np.random.randint(4))
            cumul += rew
            if done or trunc:
                break
        historique.append(cumul)
    return historique


# =============================================================================
# 4. Visualisations
# =============================================================================

def tracer_courbes_apprentissage(recompenses_ql: List[float],
                                 recompenses_aleatoire: List[float],
                                 fenetre: int = 50,
                                 chemin: str = "reports/figures/rl_courbe_apprentissage.png") -> None:
    """Courbe d'apprentissage : moyenne glissante Q-Learning vs aleatoire."""
    fig, ax = plt.subplots(figsize=(12, 6))
    moy_ql = np.convolve(recompenses_ql, np.ones(fenetre) / fenetre, mode="valid")
    moy_rand = np.convolve(recompenses_aleatoire, np.ones(fenetre) / fenetre, mode="valid")
    ax.plot(moy_ql, label="Q-Learning (moy. glissante)", color="#2196F3", linewidth=2)
    ax.plot(moy_rand, label="Politique aleatoire", color="#F44336", linewidth=2, linestyle="--")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Recompense cumulee", fontsize=12)
    ax.set_title("Apprentissage par Renforcement — Maintenance HVAC\n"
                 "Comparaison Q-Learning vs Politique Aleatoire", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Courbe sauvegardee : {chemin}")


def tracer_heatmap_q_values(agent: AgentQLearning,
                            chemin: str = "reports/figures/rl_heatmap_q_values.png") -> None:
    """Heatmap des valeurs Q pour des etats representatifs du parc HVAC."""
    # Etats representatifs : [temp, hum, age, demande, saison, budget]
    etats_cles = {
        "Ete/Neuf/Budget+":          np.array([0.7, 0.5, 0.1, 0.8, 0.8, 0.9]),
        "Ete/Vieux/Budget+":         np.array([0.7, 0.5, 0.7, 0.8, 0.8, 0.7]),
        "Hiver/Neuf/Budget+":        np.array([0.2, 0.7, 0.1, 0.7, 0.2, 0.8]),
        "Hiver/Vieux/Budget+":       np.array([0.2, 0.7, 0.7, 0.7, 0.2, 0.6]),
        "Hiver/Vieux/Budget-":       np.array([0.2, 0.7, 0.7, 0.7, 0.2, 0.2]),
        "Mi-saison/Moyen/Budget=":   np.array([0.5, 0.6, 0.4, 0.4, 0.5, 0.5]),
        "Ete/TresVieux/Budget+":     np.array([0.7, 0.5, 0.9, 0.8, 0.8, 0.6]),
        "Hiver/TresVieux/Budget-":   np.array([0.2, 0.8, 0.9, 0.9, 0.2, 0.1]),
    }
    noms_actions = ["Aucune\nmaint.", "Maint.\npreventive",
                    "Repar.\ncorrective", "Rempla-\ncement"]
    noms = list(etats_cles.keys())
    matrice = np.zeros((len(noms), 4))
    for i, obs in enumerate(etats_cles.values()):
        matrice[i], _ = agent.obtenir_q_plus_proche(obs)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrice, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(noms_actions, fontsize=10)
    ax.set_yticks(range(len(noms)))
    ax.set_yticklabels(noms, fontsize=9)
    # Annoter les cellules
    plage = matrice.max() - matrice.min() if matrice.max() != matrice.min() else 1.0
    for i in range(len(noms)):
        for j in range(4):
            c = "white" if abs(matrice[i, j]) > plage * 0.4 else "black"
            ax.text(j, i, f"{matrice[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color=c, fontweight="bold")
    ax.set_title("Valeurs Q apprises par etat et action\n"
                 "(vert = favorable, rouge = defavorable)", fontsize=13)
    fig.colorbar(im, ax=ax, label="Valeur Q")
    plt.tight_layout()
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Heatmap sauvegardee : {chemin}")


def afficher_resume_politique(agent: AgentQLearning) -> None:
    """Affiche la politique apprise pour des scenarios representatifs."""
    print("\n" + "=" * 70)
    print("RESUME DE LA POLITIQUE APPRISE")
    print("=" * 70)
    scenarios = [
        ("Equipement neuf, hiver, budget eleve",       [0.2, 0.7, 0.1, 0.7, 0.2, 0.9]),
        ("Equipement neuf, ete, budget eleve",         [0.7, 0.5, 0.1, 0.8, 0.8, 0.9]),
        ("Equipement moyen, hiver, budget moyen",      [0.2, 0.7, 0.4, 0.7, 0.2, 0.5]),
        ("Equipement vieux, hiver, budget eleve",      [0.2, 0.7, 0.7, 0.7, 0.2, 0.7]),
        ("Equipement vieux, ete, budget faible",       [0.7, 0.5, 0.7, 0.8, 0.8, 0.2]),
        ("Equipement tres vieux, hiver, budget eleve", [0.2, 0.7, 0.9, 0.9, 0.2, 0.6]),
        ("Equipement tres vieux, ete, budget faible",  [0.7, 0.5, 0.9, 0.8, 0.8, 0.1]),
    ]
    for desc, obs_list in scenarios:
        obs = np.array(obs_list, dtype=np.float32)
        q_vals, exact = agent.obtenir_q_plus_proche(obs)
        best = int(np.argmax(q_vals))
        marqueur = "" if exact else " (etat voisin)"
        print(f"\n  {desc}{marqueur}")
        print(f"    -> Action : {ACTIONS[best]}")
        print(f"    -> Q : {', '.join(f'{v:.2f}' for v in q_vals)}")
    print("\n" + "=" * 70)


# =============================================================================
# 5. Point d'entree principal
# =============================================================================

def main() -> None:
    """Entraine l'agent Q-Learning, compare a la politique aleatoire, visualise."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    N_EPISODES = 1000
    np.random.seed(42)

    print("=" * 70)
    print("APPRENTISSAGE PAR RENFORCEMENT — MAINTENANCE HVAC")
    print("Optimisation de la planification sur 96 departements francais")
    print("=" * 70)

    # --- Creation de l'environnement ---
    env = HVACMaintenanceEnv()
    print(f"\nEnvironnement : {env.n_actions} actions, 6 variables d'etat, 12 mois/episode")

    # --- Entrainement Q-Learning ---
    print(f"\n--- Entrainement Q-Learning ({N_EPISODES} episodes) ---")
    agent = AgentQLearning(n_bins=6, alpha=0.1, gamma=0.95,
                           epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995)
    recompenses_ql = entrainer_agent(env, agent, n_episodes=N_EPISODES)
    moy_ql = np.mean(recompenses_ql[-100:])
    print(f"\nQ-Learning — Moy. (100 derniers) : {moy_ql:.2f}")
    print(f"Table Q : {len(agent.q_table)} etats | Epsilon final : {agent.epsilon:.4f}")

    # --- Politique aleatoire (baseline) ---
    print(f"\n--- Politique aleatoire ({N_EPISODES} episodes) ---")
    recompenses_rand = evaluer_politique_aleatoire(env, n_episodes=N_EPISODES)
    moy_rand = np.mean(recompenses_rand)
    print(f"Aleatoire — Moy. : {moy_rand:.2f}")

    # --- Comparaison ---
    amelioration = ((moy_ql - moy_rand) / abs(moy_rand)) * 100 if moy_rand != 0 else 0
    print(f"\n--- Comparaison ---")
    print(f"  Q-Learning : {moy_ql:.2f}  |  Aleatoire : {moy_rand:.2f}  |  {amelioration:+.1f}%")

    # --- Visualisations ---
    print("\n--- Generation des visualisations ---")
    try:
        import os
        os.makedirs("reports/figures", exist_ok=True)
        tracer_courbes_apprentissage(recompenses_ql, recompenses_rand)
        tracer_heatmap_q_values(agent)
        print("Figures sauvegardees dans reports/figures/")
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder les figures : {e}")

    # --- Resume de la politique ---
    afficher_resume_politique(agent)

    # --- Note methodologique ---
    print("\n" + "-" * 70)
    print("NOTE METHODOLOGIQUE :")
    print("-" * 70)
    print("  Ce prototype utilise Q-Learning tabular, adapte a la demonstration.\n"
          "  Pour passer a l'echelle (96 departements, donnees reelles) :\n"
          "    - Stable-Baselines3 (PPO/SAC) pour du Deep RL avec PyTorch\n"
          "    - Ray RLlib pour l'entrainement distribue multi-agent\n"
          "    - Integration des donnees meteo reelles (API Meteo-France)\n"
          "    - Donnees de maintenance historiques ADEME/CSTB")
    print("-" * 70)


if __name__ == "__main__":
    main()
