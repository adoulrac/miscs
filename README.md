========================================================
FRENCH VERSION
========================================================

# GOUVERNANCE D’ACCÈS ET D’USAGE DE LA PLATEFORME
# DECISION TREE DÉTAILLÉ

========================================================
OBJECTIF DE LA PLATEFORME
========================================================

La plateforme a été créée pour répondre à un besoin réel des métiers de développer rapidement des outils tactiques, analytics et workflows, tout en réduisant les risques liés au Shadow IT non contrôlé.

L’objectif n’est PAS :
- de remplacer les plateformes IT enterprise ;
- d’héberger des systèmes critiques ;
- de contourner les standards IT ;
- de permettre la création d’applications non maintenues.

L’objectif EST :
- de réduire le Shadow IT existant ;
- de centraliser les développements tactiques ;
- d’apporter du monitoring et de la traçabilité ;
- d’apporter un cadre de gouvernance minimal ;
- de fournir des capacités LLM gouvernées ;
- de permettre l’innovation rapide des métiers ;
- de remplacer certains outils coûteux lorsque pertinent.

========================================================
PRINCIPES DE GOUVERNANCE
========================================================

1. Toute application doit avoir :
   - un responsable fonctionnel ;
   - un modèle opérationnel défini ;
   - un support identifié ;
   - un monitoring défini ;
   - une gestion des incidents définie ;
   - une responsabilité explicite du métier.

2. La plateforme fournit :
   - infrastructure ;
   - observabilité ;
   - sécurité standard ;
   - outils ;
   - intégration LLM ;
   - logging.

3. La plateforme NE fournit PAS :
   - support applicatif métier ;
   - SLA enterprise par défaut ;
   - exploitation critique ;
   - support 24/7 par défaut ;
   - ownership fonctionnel.

4. Plus la criticité augmente :
   - plus la gouvernance augmente ;
   - plus les validations augmentent ;
   - plus l’escalade vers l’IT standard devient probable.

========================================================
POPULATIONS AUTORISÉES
========================================================

1. FRONT OFFICE
- Traders
- Sales
- Structuring
- Risk
- Quant non-IT

Use cases :
- dashboards analytics ;
- outils tactiques ;
- workflow automation ;
- assistants LLM ;
- pricing tactique ;
- reporting ;
- UI métier ;
- monitoring business.

Position :
TRÈS FAVORABLE

Vigilance :
- éviter systèmes quasi-core trading ;
- éviter calculs réglementaires officiels ;
- éviter dépendances critiques ;
- éviter SLA forts ;
- éviter chaînes de décision automatiques critiques.

--------------------------------------------------------

2. OPERATIONS / MIDDLE OFFICE / BACK OFFICE

Use cases :
- automatisation workflows ;
- document processing ;
- reconciliation ;
- assistants opérationnels ;
- reporting ;
- orchestration LLM.

Position :
FAVORABLE AVEC GOUVERNANCE RENFORCÉE

Vigilance :
- données sensibles ;
- impacts opérationnels larges ;
- automatisation de décisions ;
- risques de propagation transverse.

--------------------------------------------------------

3. FINANCE / SUPPORT FUNCTIONS / AUTRES ENTITÉS

Use cases :
- dashboards ;
- reporting ;
- automatisation locale ;
- remplacement outils SaaS coûteux ;
- outils départementaux.

Position :
AUTORISÉ SI PÉRIMÈTRE MAÎTRISÉ

Vigilance :
- applications devenant critiques ;
- absence de support ;
- croissance non maîtrisée.

--------------------------------------------------------

4. DÉVELOPPEURS NON-IT OPPORTUNISTES

Use cases :
- prototypes ;
- petits outils ;
- scripts ponctuels.

Position :
AUTORISÉ UNIQUEMENT EN PÉRIMÈTRE LIMITÉ

Vigilance :
- dette opérationnelle ;
- absence de maintenance ;
- dépendance à une seule personne ;
- “prototype devenu production”.

========================================================
ARBRE DE DÉCISION COMPLET
========================================================

START
│
├── 1. Type de demande ?
│       ├── Migration application existante
│       └── Nouvelle application
│
├──────────────────────────────────────────────
│
├── A. MIGRATION APPLICATION EXISTANTE
│
│     ├── 2. L’application existe déjà hors plateforme ?
│     │       ├── Oui
│     │       └── Non → Branche Nouvelle Application
│     │
│     ├── 3. L’application est utilisée activement ?
│     │       ├── Usage quotidien
│     │       ├── Usage régulier
│     │       ├── Usage départemental
│     │       ├── Usage multi-entités
│     │       └── Peu utilisée
│     │
│     ├── 4. Où tourne-t-elle ?
│     │       ├── Laptop utilisateur
│     │       ├── Excel/VBA
│     │       ├── Script Python local
│     │       ├── VM non gouvernée
│     │       ├── Serveur départemental
│     │       ├── Infrastructure non standard
│     │       └── Outil tiers non contrôlé
│     │
│     │       └── Migration fortement encouragée
│     │
│     ├── 5. Population métier ?
│     │       ├── Front Office
│     │       ├── Operations
│     │       ├── Finance
│     │       ├── Support Functions
│     │       └── Other
│     │
│     ├── 6. Use case ?
│     │       ├── Dashboard
│     │       ├── Analytics
│     │       ├── Workflow automation
│     │       ├── Reporting
│     │       ├── Assistant LLM
│     │       ├── Pricing tactique
│     │       ├── Monitoring business
│     │       ├── UI métier
│     │       └── Calcul officiel
│     │
│     ├── 7. Niveau de criticité ?
│     │       ├── Faible
│     │       ├── Moyenne
│     │       ├── Élevée
│     │       └── Critique
│     │
│     ├── 8. Si critique :
│     │       ├── SLA fort ?
│     │       ├── Support 24/7 ?
│     │       ├── Haute disponibilité ?
│     │       ├── Temps réel critique ?
│     │       ├── Impact financier ?
│     │       ├── Usage réglementaire ?
│     │       ├── Décision automatisée ?
│     │       ├── Dépendance business forte ?
│     │       ├── Processus officiel banque ?
│     │       └── Enterprise-wide ?
│     │
│     │       ├── Oui → Governance / Architecture Escalation
│     │       └── Non → Validation conditionnelle
│     │
│     ├── 9. Données manipulées ?
│     │       ├── Données internes
│     │       ├── Données sensibles
│     │       ├── Données clients
│     │       ├── Données réglementées
│     │       ├── Données trading sensibles
│     │       └── Documents sensibles
│     │
│     ├── 10. Usage LLM ?
│     │       ├── Oui
│     │       │
│     │       ├── Provider approuvé ?
│     │       ├── Logging activé ?
│     │       ├── Prompt monitoring activé ?
│     │       ├── Data retention conforme ?
│     │       ├── Données sensibles envoyées ?
│     │       └── AI Governance Review
│     │
│     │       └── Non
│     │
│     ├── 11. Modèle opérationnel défini ?
│     │       ├── Support identifié
│     │       ├── Monitoring défini
│     │       ├── Alerting défini
│     │       ├── Incident management défini
│     │       ├── Maintenance définie
│     │       ├── Responsable fonctionnel identifié
│     │       ├── Documentation minimale
│     │       └── Gestion lifecycle définie
│     │
│     │       ├── Oui → APPROVED
│     │       └── Non → REFUSED TEMPORARILY
│
├──────────────────────────────────────────────
│
├── B. NOUVELLE APPLICATION
│
│     ├── 12. Population métier ?
│     │       ├── Front Office
│     │       ├── Operations
│     │       ├── Finance
│     │       ├── Support Functions
│     │       └── Other
│     │
│     ├── 13. Objectif principal ?
│     │       ├── Réduction Shadow IT
│     │       ├── Productivité
│     │       ├── Innovation tactique
│     │       ├── Workflow automation
│     │       ├── Remplacement outil coûteux
│     │       ├── Reporting
│     │       ├── Dashboarding
│     │       ├── Assistant LLM
│     │       └── Construction système stratégique
│     │
│     │       ├── Système stratégique → REFUSED
│     │       └── Usage tactique
│     │
│     ├── 14. Scope attendu ?
│     │       ├── Individuel
│     │       ├── Équipe locale
│     │       ├── Département
│     │       ├── Multi-entités
│     │       └── Enterprise-wide
│     │
│     │       ├── Enterprise-wide → Architecture Review
│     │       └── Scope limité
│     │
│     ├── 15. Niveau de criticité ?
│     │       ├── Faible
│     │       ├── Moyenne
│     │       ├── Élevée
│     │       └── Critique
│     │
│     ├── 16. Besoins techniques ?
│     │       ├── Temps réel critique
│     │       ├── SLA fort
│     │       ├── Haute disponibilité
│     │       ├── Support 24/7
│     │       ├── Résilience forte
│     │       └── Gros volumes critiques
│     │
│     │       ├── Oui → IT Governance Escalation
│     │       └── Non
│     │
│     ├── 17. Données manipulées ?
│     │       ├── Données simples
│     │       ├── Données sensibles
│     │       ├── Données clients
│     │       ├── Données réglementées
│     │       ├── Données marchés sensibles
│     │       └── Documents confidentiels
│     │
│     ├── 18. Usage LLM ?
│     │       ├── Oui
│     │       │
│     │       ├── Provider approuvé ?
│     │       ├── Monitoring activé ?
│     │       ├── Logging activé ?
│     │       ├── Contrôle prompts/réponses ?
│     │       ├── Protection données sensibles ?
│     │       └── AI Governance Validation
│     │
│     │       └── Non
│     │
│     ├── 19. Niveau d’autonomie ?
│     │       ├── Assisté utilisateur
│     │       ├── Semi-automatisé
│     │       └── Décision automatique
│     │
│     │       ├── Décision automatique → Risk Escalation
│     │       └── Assisté
│     │
│     ├── 20. Modèle opérationnel défini ?
│     │       ├── Support identifié
│     │       ├── Monitoring défini
│     │       ├── Alerting défini
│     │       ├── Incident management défini
│     │       ├── Maintenance définie
│     │       ├── Lifecycle management défini
│     │       ├── Documentation minimale
│     │       └── Responsable fonctionnel identifié
│     │
│     │       ├── Oui
│     │       └── Non → REFUSED
│     │
│     └── 21. DÉCISION FINALE
│             ├── Faible risque → AUTO-APPROVAL
│             ├── Risque modéré → PLATFORM REVIEW
│             ├── Risque élevé → GOVERNANCE REVIEW
│             ├── Cas critique → EXECUTIVE ESCALATION
│             └── Usage incompatible → REFUSED

========================================================
ENGLISH VERSION
========================================================

# PLATFORM ACCESS & USAGE GOVERNANCE
# DETAILED DECISION TREE

========================================================
PLATFORM OBJECTIVE
========================================================

The platform was created to address a real business need for rapidly developing tactical tools, analytics and workflows while reducing risks associated with unmanaged Shadow IT.

The platform is NOT intended:
- to replace enterprise IT platforms;
- to host critical systems;
- to bypass IT standards;
- to allow unmanaged applications.

The platform IS intended:
- to reduce existing Shadow IT;
- to centralize tactical developments;
- to provide monitoring and traceability;
- to introduce lightweight governance;
- to provide governed LLM capabilities;
- to accelerate business innovation;
- to replace expensive tools when relevant.

========================================================
GOVERNANCE PRINCIPLES
========================================================

1. Every application must have:
   - a functional owner;
   - a defined operational model;
   - identified support;
   - defined monitoring;
   - incident management;
   - explicit business accountability.

2. The platform provides:
   - infrastructure;
   - observability;
   - standard security;
   - tooling;
   - LLM integration;
   - logging.

3. The platform does NOT provide:
   - business application support;
   - enterprise SLA by default;
   - critical operations support;
   - default 24/7 support;
   - functional ownership.

4. The higher the criticality:
   - the stronger the governance;
   - the stronger the validation requirements;
   - the higher the probability of escalation to standard IT.

========================================================
AUTHORIZED POPULATIONS
========================================================

1. FRONT OFFICE
- Traders
- Sales
- Structuring
- Risk
- Non-IT Quants

Typical use cases:
- analytics dashboards;
- tactical tools;
- workflow automation;
- LLM assistants;
- tactical pricing;
- reporting;
- business UIs;
- business monitoring.

Position:
HIGHLY FAVORABLE

Watchpoints:
- avoid quasi-core trading systems;
- avoid official regulatory calculations;
- avoid critical dependencies;
- avoid strong SLA expectations;
- avoid critical automated decision chains.

--------------------------------------------------------

2. OPERATIONS / MIDDLE OFFICE / BACK OFFICE

Use cases:
- workflow automation;
- document processing;
- reconciliation;
- operational assistants;
- reporting;
- LLM orchestration.

Position:
FAVORABLE WITH REINFORCED GOVERNANCE

Watchpoints:
- sensitive data;
- large operational impacts;
- automated decisions;
- cross-functional propagation risks.

--------------------------------------------------------

3. FINANCE / SUPPORT FUNCTIONS / OTHER ENTITIES

Use cases:
- dashboards;
- reporting;
- local automation;
- replacement of expensive SaaS tools;
- departmental applications.

Position:
ALLOWED IF SCOPE REMAINS CONTROLLED

Watchpoints:
- applications becoming critical;
- lack of support;
- uncontrolled growth.

--------------------------------------------------------

4. OPPORTUNISTIC NON-IT DEVELOPERS

Use cases:
- prototypes;
- small tools;
- ad hoc scripts.

Position:
ALLOWED ONLY IN LIMITED SCOPE

Watchpoints:
- operational debt;
- lack of maintenance;
- dependency on one person;
- “prototype becoming production”.

========================================================
FULL DECISION TREE
========================================================

START
│
├── 1. Request type?
│       ├── Existing application migration
│       └── New application
│
├──────────────────────────────────────────────
│
├── A. EXISTING APPLICATION MIGRATION
│
│     ├── 2. Does the application already exist outside the platform?
│     │       ├── Yes
│     │       └── No → Switch to New Application branch
│     │
│     ├── 3. Is the application actively used?
│     │       ├── Daily usage
│     │       ├── Regular usage
│     │       ├── Departmental usage
│     │       ├── Multi-entity usage
│     │       └── Low usage
│     │
│     ├── 4. Where is the application currently running?
│     │       ├── User laptop
│     │       ├── Excel/VBA
│     │       ├── Local Python script
│     │       ├── Unmanaged VM
│     │       ├── Departmental server
│     │       ├── Non-standard infrastructure
│     │       └── Uncontrolled third-party tool
│     │
│     │       └── Migration strongly encouraged
│     │
│     ├── 5. Business population?
│     │       ├── Front Office
│     │       ├── Operations
│     │       ├── Finance
│     │       ├── Support Functions
│     │       └── Other
│     │
│     ├── 6. Use case?
│     │       ├── Dashboard
│     │       ├── Analytics
│     │       ├── Workflow automation
│     │       ├── Reporting
│     │       ├── LLM Assistant
│     │       ├── Tactical pricing
│     │       ├── Business monitoring
│     │       ├── Business UI
│     │       └── Official calculation
│     │
│     ├── 7. Criticality level?
│     │       ├── Low
│     │       ├── Medium
│     │       ├── High
│     │       └── Critical
│     │
│     ├── 8. If critical:
│     │       ├── Strong SLA required?
│     │       ├── 24/7 support required?
│     │       ├── High availability required?
│     │       ├── Critical real-time processing?
│     │       ├── Financial impact?
│     │       ├── Regulatory usage?
│     │       ├── Automated decision making?
│     │       ├── Strong business dependency?
│     │       ├── Official bank process?
│     │       └── Enterprise-wide?
│     │
│     │       ├── Yes → Governance / Architecture Escalation
│     │       └── No → Conditional approval
│     │
│     ├── 9. Data processed?
│     │       ├── Internal data
│     │       ├── Sensitive data
│     │       ├── Client data
│     │       ├── Regulated data
│     │       ├── Sensitive trading data
│     │       └── Sensitive documents
│     │
│     ├── 10. LLM usage?
│     │       ├── Yes
│     │       │
│     │       ├── Approved provider?
│     │       ├── Logging enabled?
│     │       ├── Prompt monitoring enabled?
│     │       ├── Compliant data retention?
│     │       ├── Sensitive data transmitted?
│     │       └── AI Governance Review
│     │
│     │       └── No
│     │
│     ├── 11. Operational model defined?
│     │       ├── Support identified
│     │       ├── Monitoring defined
│     │       ├── Alerting defined
│     │       ├── Incident management defined
│     │       ├── Maintenance defined
│     │       ├── Functional owner identified
│     │       ├── Minimal documentation
│     │       └── Lifecycle management defined
│     │
│     │       ├── Yes → APPROVED
│     │       └── No → REFUSED TEMPORARILY
│
├──────────────────────────────────────────────
│
├── B. NEW APPLICATION
│
│     ├── 12. Business population?
│     │       ├── Front Office
│     │       ├── Operations
│     │       ├── Finance
│     │       ├── Support Functions
│     │       └── Other
│     │
│     ├── 13. Main objective?
│     │       ├── Shadow IT reduction
│     │       ├── Productivity improvement
│     │       ├── Tactical innovation
│     │       ├── Workflow automation
│     │       ├── Expensive tool replacement
│     │       ├── Reporting
│     │       ├── Dashboarding
│     │       ├── LLM Assistant
│     │       └── Strategic system development
│     │
│     │       ├── Strategic system → REFUSED
│     │       └── Tactical usage
│     │
│     ├── 14. Expected scope?
│     │       ├── Individual
│     │       ├── Local team
│     │       ├── Department
│     │       ├── Multi-entity
│     │       └── Enterprise-wide
│     │
│     │       ├── Enterprise-wide → Architecture Review
│     │       └── Limited scope
│     │
│     ├── 15. Criticality level?
│     │       ├── Low
│     │       ├── Medium
│     │       ├── High
│     │       └── Critical
│     │
│     ├── 16. Technical requirements?
│     │       ├── Critical real-time processing
│     │       ├── Strong SLA
│     │       ├── High availability
│     │       ├── 24/7 support
│     │       ├── Strong resilience
│     │       └── Critical large-scale processing
│     │
│     │       ├── Yes → IT Governance Escalation
│     │       └── No
│     │
│     ├── 17. Data processed?
│     │       ├── Simple data
│     │       ├── Sensitive data
│     │       ├── Client data
│     │       ├── Regulated data
│     │       ├── Sensitive market data
│     │       └── Confidential documents
│     │
│     ├── 18. LLM usage?
│     │       ├── Yes
│     │       │
│     │       ├── Approved provider?
│     │       ├── Monitoring enabled?
│     │       ├── Logging enabled?
│     │       ├── Prompt/response control?
│     │       ├── Sensitive data protection?
│     │       └── AI Governance Validation
│     │
│     │       └── No
│     │
│     ├── 19. Autonomy level?
│     │       ├── User-assisted
│     │       ├── Semi-automated
│     │       └── Fully automated decision making
│     │
│     │       ├── Automated decision making → Risk Escalation
│     │       └── Assisted
│     │
│     ├── 20. Operational model defined?
│     │       ├── Support identified
│     │       ├── Monitoring defined
│     │       ├── Alerting defined
│     │       ├── Incident management defined
│     │       ├── Maintenance defined
│     │       ├── Lifecycle management defined
│     │       ├── Minimal documentation
│     │       └── Functional owner identified
│     │
│     │       ├── Yes
│     │       └── No → REFUSED
│     │
│     └── 21. FINAL DECISION
│             ├── Low risk → AUTO-APPROVAL
│             ├── Medium risk → PLATFORM REVIEW
│             ├── High risk → GOVERNANCE REVIEW
│             ├── Critical case → EXECUTIVE ESCALATION
│             └── Incompatible usage → REFUSED









========================================================
VERSION FRANÇAISE
========================================================

# GOUVERNANCE D’ACCÈS ET D’USAGE DE LA PLATEFORME
# DECISION TREE DÉTAILLÉ

========================================================
OBJECTIF DE LA PLATEFORME
========================================================

La plateforme a été créée pour répondre à un besoin réel des métiers de développer rapidement des outils tactiques, analytics et workflows, tout en réduisant les risques liés au Shadow IT non contrôlé.

L’objectif n’est PAS :
- de remplacer les plateformes IT enterprise ;
- d’héberger des systèmes critiques ;
- de contourner les standards IT ;
- de permettre la création d’applications non maintenues.

L’objectif EST :
- de réduire le Shadow IT existant ;
- de centraliser les développements tactiques ;
- d’apporter du monitoring et de la traçabilité ;
- d’apporter un cadre de gouvernance minimal ;
- de fournir des capacités LLM gouvernées ;
- de permettre l’innovation rapide des métiers ;
- de remplacer certains outils coûteux lorsque pertinent.

========================================================
PRINCIPES DE GOUVERNANCE
========================================================

1. Toute application doit avoir :
   - un responsable fonctionnel ;
   - un modèle opérationnel défini ;
   - un support identifié ;
   - un monitoring défini ;
   - une gestion des incidents définie ;
   - une responsabilité explicite du métier.

2. La plateforme fournit :
   - infrastructure ;
   - observabilité ;
   - sécurité standard ;
   - outils ;
   - intégration LLM ;
   - logging.

3. La plateforme NE fournit PAS :
   - support applicatif métier ;
   - SLA enterprise par défaut ;
   - exploitation critique ;
   - support 24/7 par défaut ;
   - ownership fonctionnel.

4. Plus la criticité augmente :
   - plus la gouvernance augmente ;
   - plus les validations augmentent ;
   - plus l’escalade vers l’IT standard devient probable.

========================================================
POPULATIONS AUTORISÉES
========================================================

--------------------------------------------------------
1. FRONT OFFICE
--------------------------------------------------------

Profils :
- Traders
- Sales
- Structuring
- Risk
- Quant non-IT

Use cases typiques :
- dashboards analytics ;
- outils tactiques ;
- workflow automation ;
- assistants LLM ;
- pricing tactique ;
- reporting ;
- UI métier ;
- monitoring business.

Position :
TRÈS FAVORABLE

Raison :
- historique fort de Shadow IT ;
- besoin de rapidité ;
- valeur business élevée ;
- alternative plus sécurisée au local.

Vigilance :
- éviter systèmes quasi-core trading ;
- éviter calculs réglementaires officiels ;
- éviter dépendances critiques ;
- éviter SLA forts ;
- éviter chaînes de décision automatiques critiques.

--------------------------------------------------------
2. OPERATIONS / MIDDLE OFFICE / BACK OFFICE
--------------------------------------------------------

Use cases :
- automatisation workflows ;
- document processing ;
- reconciliation ;
- assistants opérationnels ;
- reporting ;
- orchestration LLM.

Position :
FAVORABLE AVEC GOUVERNANCE RENFORCÉE

Vigilance :
- données sensibles ;
- impacts opérationnels larges ;
- automatisation de décisions ;
- risques de propagation transverse.

--------------------------------------------------------
3. FINANCE / SUPPORT FUNCTIONS / AUTRES ENTITÉS
--------------------------------------------------------

Use cases :
- dashboards ;
- reporting ;
- automatisation locale ;
- remplacement outils SaaS coûteux ;
- outils départementaux.

Position :
AUTORISÉ SI PÉRIMÈTRE MAÎTRISÉ

Vigilance :
- applications devenant critiques ;
- absence de support ;
- croissance non maîtrisée.

--------------------------------------------------------
4. DÉVELOPPEURS NON-IT OPPORTUNISTES
--------------------------------------------------------

Use cases :
- prototypes ;
- petits outils ;
- scripts ponctuels.

Position :
AUTORISÉ UNIQUEMENT EN PÉRIMÈTRE LIMITÉ

Vigilance :
- dette opérationnelle ;
- absence de maintenance ;
- dépendance à une seule personne ;
- “prototype devenu production”.

========================================================
ARBRE DE DÉCISION COMPLET
========================================================

START
│
├── 1. Type de demande ?
│       │
│       ├── A. Migration application existante
│       │
│       └── B. Nouvelle application
│
├──────────────────────────────────────────────
│
├── A. MIGRATION APPLICATION EXISTANTE
│
│     ├── 2. L’application existe-t-elle déjà hors plateforme ?
│     │       │
│     │       ├── Oui
│     │       │
│     │       └── Non
│     │              └── Basculer vers branche Nouvelle Application
│     │
│     ├── 3. L’application est-elle déjà utilisée activement ?
│     │       │
│     │       ├── Usage quotidien
│     │       ├── Usage régulier
│     │       ├── Usage départemental
│     │       ├── Usage multi-entités
│     │       └── Peu utilisée
│     │
│     ├── 4. Où tourne actuellement l’application ?
│     │       │
│     │       ├── Laptop utilisateur
│     │       ├── Excel/VBA
│     │       ├── Script Python local
│     │       ├── VM non gouvernée
│     │       ├── Serveur départemental
│     │       ├── Infrastructure non standard
│     │       └── Outil tiers non contrôlé
│     │
│     │       ├── Si oui
│     │       │      └── Migration fortement encouragée
│     │       │
│     │       └── Sinon
│     │              └── Continuer analyse
│     │
│     ├── 5. Population métier ?
│     │       │
│     │       ├── Front Office
│     │       ├── Operations
│     │       ├── Finance
│     │       ├── Support Functions
│     │       └── Other
│     │
│     ├── 6. Type de use case ?
│     │       │
│     │       ├── Dashboard
│     │       ├── Analytics
│     │       ├── Workflow automation
│     │       ├── Reporting
│     │       ├── Assistant LLM
│     │       ├── Pricing tactique
│     │       ├── Monitoring business
│     │       ├── UI métier
│     │       └── Calcul officiel
│     │
│     ├── 7. Niveau de criticité ?
│     │       │
│     │       ├── Faible
│     │       ├── Moyenne
│     │       ├── Élevée
│     │       └── Critique
│     │
│     ├── 8. Si critique :
│     │       │
│     │       ├── SLA fort requis ?
│     │       ├── Support 24/7 requis ?
│     │       ├── Haute disponibilité requise ?
│     │       ├── Temps réel critique ?
│     │       ├── Impact financier direct ?
│     │       ├── Usage réglementaire ?
│     │       ├── Décision automatisée ?
│     │       ├── Dépendance opérationnelle majeure ?
│     │       ├── Processus officiel banque ?
│     │       └── Application enterprise-wide ?
│     │
│     │       ├── Oui à un ou plusieurs
│     │       │      └── Governance / Architecture Escalation
│     │       │
│     │       └── Non
│     │              └── Validation conditionnelle
│     │
│     ├── 9. Données manipulées ?
│     │       │
│     │       ├── Données internes simples
│     │       ├── Données confidentielles
│     │       ├── Données clients
│     │       ├── Données réglementées
│     │       ├── Données trading sensibles
│     │       └── Documents sensibles
│     │
│     ├── 10. Usage LLM ?
│     │       │
│     │       ├── Oui
│     │       │
│     │       ├── Provider approuvé ?
│     │       ├── Logging activé ?
│     │       ├── Prompt monitoring activé ?
│     │       ├── Data retention conforme ?
│     │       ├── Données sensibles envoyées ?
│     │       └── AI Governance Review
│     │       │
│     │       └── Non
│     │
│     ├── 11. Modèle opérationnel défini ?
│     │       │
│     │       ├── Support identifié
│     │       ├── Monitoring défini
│     │       ├── Alerting défini
│     │       ├── Incident management défini
│     │       ├── Maintenance définie
│     │       ├── Responsable fonctionnel identifié
│     │       ├── Documentation minimale
│     │       └── Gestion lifecycle définie
│     │
│     │       ├── Oui
│     │       │      └── APPROVED
│     │       │
│     │       └── Non
│     │              └── REFUSED TEMPORARILY
│
├──────────────────────────────────────────────
│
├── B. NOUVELLE APPLICATION
│
│     ├── 12. Population métier ?
│     │       │
│     │       ├── Front Office
│     │       ├── Operations
│     │       ├── Finance
│     │       ├── Support Functions
│     │       └── Other
│     │
│     ├── 13. Objectif principal ?
│     │       │
│     │       ├── Réduction Shadow IT
│     │       ├── Productivité
│     │       ├── Innovation tactique
│     │       ├── Workflow automation
│     │       ├── Remplacement outil coûteux
│     │       ├── Reporting
│     │       ├── Dashboarding
│     │       ├── Assistant LLM
│     │       └── Construction système stratégique
│     │
│     │       ├── Construction système stratégique
│     │       │      └── REFUSED / Redirect Standard IT
│     │       │
│     │       └── Usage tactique
│     │
│     ├── 14. Scope attendu ?
│     │       │
│     │       ├── Individuel
│     │       ├── Équipe locale
│     │       ├── Département
│     │       ├── Multi-entités
│     │       └── Enterprise-wide
│     │
│     │       ├── Enterprise-wide
│     │       │      └── Mandatory Architecture Review
│     │       │
│     │       └── Scope limité
│     │
│     ├── 15. Niveau de criticité ?
│     │       │
│     │       ├── Faible
│     │       ├── Moyenne
│     │       ├── Élevée
│     │       └── Critique
│     │
│     ├── 16. Besoins techniques ?
│     │       │
│     │       ├── Temps réel critique
│     │       ├── SLA fort
│     │       ├── Haute disponibilité
│     │       ├── Support 24/7
│     │       ├── Résilience forte
│     │       └── Gros volumes critiques
│     │
│     │       ├── Oui
│     │       │      └── IT Governance Escalation
│     │       │
│     │       └── Non
│     │
│     ├── 17. Données manipulées ?
│     │       │
│     │       ├── Données simples
│     │       ├── Données sensibles
│     │       ├── Données clients
│     │       ├── Données réglementées
│     │       ├── Données marchés sensibles
│     │       └── Documents confidentiels
│     │
│     ├── 18. Usage LLM ?
│     │       │
│     │       ├── Oui
│     │       │
│     │       ├── Provider approuvé ?
│     │       ├── Monitoring activé ?
│     │       ├── Logging activé ?
│     │       ├── Contrôle prompts/réponses ?
│     │       ├── Protection données sensibles ?
│     │       └── AI Governance Validation
│     │       │
│     │       └── Non
│     │
│     ├── 19. Niveau d’autonomie ?
│     │       │
│     │       ├── Assisté utilisateur
│     │       ├── Semi-automatisé
│     │       └── Décision automatique
│     │
│     │       ├── Décision automatique
│     │       │      └── Risk Escalation
│     │       │
│     │       └── Assisté
│     │
│     ├── 20. Modèle opérationnel défini ?
│     │       │
│     │       ├── Support identifié
│     │       ├── Monitoring défini
│     │       ├── Alerting défini
│     │       ├── Incident management défini
│     │       ├── Maintenance définie
│     │       ├── Lifecycle management défini
│     │       ├── Documentation minimale
│     │       └── Responsable fonctionnel identifié
│     │
│     │       ├── Oui
│     │       │
│     │       └── Non
│     │              └── REFUSED
│     │
│     └── 21. DÉCISION FINALE
│             │
│             ├── Faible risque
│             │      └── AUTO-APPROVAL
│             │
│             ├── Risque modéré
│             │      └── PLATFORM REVIEW
│             │
│             ├── Risque élevé
│             │      └── GOVERNANCE REVIEW
│             │
│             ├── Cas critique
│             │      └── EXECUTIVE ESCALATION
│             │
│             └── Usage incompatible
│                    └── REFUSED

========================================================
ENGLISH VERSION
========================================================

# PLATFORM ACCESS & USAGE GOVERNANCE
# DETAILED DECISION TREE

[Same structure translated in English can follow here if needed]





















Great, thanks for the detailed overview — this gives me a full picture of your scope, and it’s impressive. You’re overseeing four very complementary teams that cover a wide landscape: data, surveillance, front office tools, and strategic platforms.

Given your goal — a first-round, non-technical but strategic screening — I’ll break this down into:

⸻

1. Updated Short Presentation (to give to candidates)

(Updated with your four teams – feel free to copy/paste or modify)

⸻

BNP Paribas CIB – Global Markets IT – E-Trading / Data & Platform Projects – VIE Role in New York

Hello! I’m an IT Manager based in New York, working within BNP Paribas Corporate & Institutional Banking (CIB), in the Global Markets IT department, leading several teams across data, trade surveillance, front office tools, and platform development.

You’ll be joining a rich and dynamic environment at the heart of BNP Paribas’ Global Markets activity. My teams include:
	1.	Global Markets Data Warehouse (GMDWH)
A transversal Oracle-based data warehouse centralizing pre-trade flows across all asset classes. Used for regulatory, compliance, surveillance, and front office purposes.
	2.	Sonata
A cross-asset trade surveillance team developing detection and analytics tools across commodities, equities and derivatives, fixed income (credit & rates), and FX.
	3.	Equities & Derivatives Front Office IT
A team located in the US working directly with equity traders on tools like option pricing, volatility computation, and risk metrics (delta, gamma, vega, etc.).
	4.	FinStrat Platform
A cross-regional strategic platform (with teams in NY, Paris, and London) enabling non-IT users to create robust, scalable, and compliant applications — avoiding shadow IT. It’s used across global business lines and touches every asset class.

As a VIE, you’ll support transversal initiatives and may get involved with one or more of these teams depending on your interest, background, and evolution.

⸻

2. VIE Candidate Evaluation: Full List of Interview Questions

Here’s a comprehensive bank of questions, categorized by themes, to help you get the most out of the interview. These questions will help you assess motivation, curiosity, communication, fit for each team, and general mindset.

⸻

A. General Motivation & Adaptability (all teams)
	•	What motivated you to apply for a VIE in New York, specifically within Global Markets IT?
	•	How do you feel about working in a fast-paced, multi-cultural environment?
	•	What do you expect from your VIE experience, personally and professionally?
	•	Have you ever worked on projects with multiple teams or time zones? How did you adapt?
	•	How do you approach situations where you’re outside your comfort zone?

⸻

B. Communication & Stakeholder Management
	•	Tell me about a time you had to explain a technical concept to a non-technical stakeholder.
	•	How do you ensure you’re aligned with stakeholders’ expectations on a project?
	•	How would you react if a trader asked you for an urgent fix or feature you don’t yet understand?

⸻

C. Curiosity & Functional Interest (fit for GMDWH / Sonata / FinStrat)
	•	What do you know about trade surveillance or data quality in banking?
	•	Have you ever worked with or manipulated financial data? How did you approach it?
	•	Are there particular asset classes or business lines that interest you more (equities, FX, credit, commodities, etc.)?
	•	Do you follow financial news or markets? Can you share a story or trend that caught your eye recently?
	•	Would you be more comfortable writing SQL queries, analyzing trade data, or designing tools for users? Why?

⸻

D. Platform Mindset & Problem Solving (fit for FinStrat)
	•	Have you ever supported a platform or tool used by others? How did you handle user support or incidents?
	•	When building a reusable solution, what’s important to you — performance, UX, robustness? Why?
	•	How would you deal with a user who created a broken application on the platform and blames the platform?

⸻

E. Analytical Thinking & Flexibility (fit for Sonata / Front Office IT)
	•	If you’re given an Excel file with thousands of trades and asked to find “weird patterns,” what’s your process?
	•	Do you enjoy digging into code, understanding data, or talking with users? Which one do you prefer and why?
	•	Would you prefer working on one deep problem or switching frequently between smaller tasks?

⸻

F. Fit for the Front Office IT (Equities & Derivatives)
	•	Do you have any familiarity with options or derivatives? (Even theoretical.)
	•	If I ask you to explain what “volatility” means in finance, how would you do it?
	•	Do you enjoy working closely with traders and solving real-time issues?
	•	Are you interested in mathematical modeling or market behavior?

⸻

G. Ownership & Learning Attitude
	•	Tell me about a project or tool you’re proud of having contributed to. What was your role?
	•	When you face something you don’t understand (business logic, code, data), what’s your strategy?
	•	How do you stay up-to-date with technology or finance?

⸻

H. Role Matching & Self-Assessment

Ask this at the end to get their self-positioning and help you decide which team they’d best fit into:
	•	Based on everything we’ve discussed, which of the four teams do you think you’d be most excited to work with, and why?
	•	Do you see yourself more as a platform person, a data person, or someone who enjoys being closer to trading activity?

⸻

3. Tips to Challenge Them & Spot the Best
	•	Give them a real-life scenario:
“A front office user is complaining that the data looks off. You’re not sure why. What steps would you take?” — This lets you see their reasoning, not just their answers.
	•	Ask for counter-questions:
A curious candidate will ask you questions. Ask:
“Do you have any questions for me about the teams, projects, or challenges?”
Lack of questions is often a red flag.
	•	Make it conversational, not rigid:
You want to assess not only knowledge or attitude but also how they would integrate into a collaborative and dynamic setup like yours.
	•	Don’t be afraid to challenge gently:
“You said you’re interested in markets. What was the last thing you read about rates or equities?”
This shows how deep their interest really is.

⸻

Let me know if you’d like:
	•	A 1-page PDF version of the team presentation
	•	A scorecard template to evaluate each candidate
	•	Help drafting the internal VIE offer document

Happy to help make this VIE search a success.


Perfect — here’s a shorter, sharper version of your presentation + a finance-focused question bank with answers. This keeps things engaging, shows the scope, and tests functional readiness without being overly technical.

⸻

Short Intro Presentation – BNP Paribas Global Markets IT / E-Trading – VIE in NYC

At BNP Paribas, the Global Markets IT department supports all front office activities across trading, sales, and structuring teams. We’re embedded within Global Markets, and provide tools, data, platforms, and analytics across all asset classes — from equities and FX to rates, credit, and commodities.

I’m part of the E-Trading division, which focuses on data, platforms, and electronic trading services. I manage four transversal teams:
	1.	Global Markets Data Warehouse (GMDWH):
Centralizes pre-trade data from all trading systems across asset classes. Used for compliance, surveillance, reporting, and front office analytics.
	2.	Sonata – Surveillance Applications:
Builds cross-asset trade surveillance tools to detect suspicious patterns or behaviors (e.g. spoofing, layering). Functional coverage includes commodities, equities, derivatives, FX, credit, and rates.
	3.	Equities & Derivatives Front Office IT:
US-based team that supports equity derivatives traders with tools for option pricing, volatility, greeks (delta, gamma, etc.), and intraday analytics.
	4.	FinStrat Platform:
A strategic platform enabling front office teams to build scalable, IT-compliant applications — removing shadow IT. It’s cross-asset and user-facing, and requires a mix of dev skills, curiosity, and business support.

As a VIE, you may be involved in one or several of these teams based on your skills and evolution.

⸻

Finance & Front Office – Functional Questions (With Answers)

These questions are designed to evaluate core front office knowledge, market awareness, and functional curiosity.

⸻

A. Market Structure & Front Office Basics

Q1. What’s the difference between an order and a trade?
A:
	•	An order is an intention to buy or sell at a certain price and quantity.
	•	A trade is the actual execution of that order. You can have many trades from one order (e.g. partial fills).

⸻

Q2. What is a position?
A:
A position is the net quantity of a financial instrument a trader holds.
	•	Long position = you bought more than you sold.
	•	Short position = you sold more than you bought.

⸻

Q3. What is P&L and what are its components?
A:
P&L = Profit and Loss, the financial result of trading. It includes:
	•	Realized P&L: From trades already executed and closed.
	•	Unrealized P&L: From current open positions, based on market price.
	•	Total P&L = Realized + Unrealized.

⸻

Q4. What are greeks in options trading?
A:
Greeks measure sensitivity of an option’s price to market variables:
	•	Delta: sensitivity to underlying price
	•	Gamma: sensitivity of delta
	•	Vega: sensitivity to volatility
	•	Theta: sensitivity to time decay
	•	Rho: sensitivity to interest rates

⸻

Q5. What is volatility?
A:
Volatility measures how much the price of an asset fluctuates over time.
	•	Historical volatility: based on past prices
	•	Implied volatility: market’s forecast of future volatility, derived from option prices.

⸻

Q6. What is market making?
A:
A market maker provides continuous bid and ask quotes to ensure liquidity. They profit from the bid-ask spread and manage their risk using hedging strategies.

⸻

Q7. What’s the difference between cash and derivatives products?
A:
	•	Cash products: like stocks or bonds, represent ownership or debt.
	•	Derivatives: like options or futures, derive their value from an underlying asset.

⸻

B. Asset Class Questions

Q8. What are the main asset classes in a trading floor?
A:
	•	Equities: stocks and indices
	•	Fixed Income: bonds, rates, credit
	•	FX: currencies
	•	Commodities: oil, metals, agriculture
	•	Derivatives: options, futures, swaps — exist across all asset classes

⸻

Q9. What’s a CDS (Credit Default Swap)?
A:
A CDS is a derivative used to insure against the default of a borrower. It’s like buying insurance on a bond — if the issuer defaults, the seller pays you.

⸻

Q10. What is the yield of a bond?
A:
Yield is the effective return of a bond, usually expressed annually.
	•	Current yield = coupon / market price
	•	Yield to maturity considers all future cash flows and current price.

⸻

C. Electronic Trading & Surveillance Context

Q11. What is algorithmic trading?
A:
Using algorithms to execute orders automatically based on pre-defined rules (e.g., price, volume, time). Often used in high-frequency or low-latency contexts.

⸻

Q12. What is spoofing in trading?
A:
A market manipulation technique where a trader places large fake orders to influence price, then cancels them after executing on the other side. It’s illegal and closely monitored in surveillance.

⸻

Q13. What kind of data is stored in a pre-trade data warehouse?
A:
Examples include:
	•	Orders and quotes
	•	Prices and market data
	•	Trader decisions
	•	Algorithm parameters
Used for surveillance, compliance, reporting, and business analytics.

⸻

Q14. What’s the difference between latency and throughput in electronic trading?
A:
	•	Latency: how fast something happens (e.g., time to send an order)
	•	Throughput: how much data can be handled over time (e.g., orders/sec)

⸻

Optional: Ask the candidate to explain one of these concepts to you

This is a great way to test understanding and clarity.

⸻

Let me know if you’d like me to bundle this into a 1-page candidate briefing, or generate a scorecard template to keep interview evaluations consistent.