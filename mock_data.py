# mock_data.py
# Simulates the output of your RAG pipeline:
#   - question: user query
#   - contexts: chunks retrieved from Weaviate (the "source of truth")
#   - answer: what your LLM app responded
#   - ground_truth: ideal reference answer (used for context_recall metric)

MOCK_DATASET = [
    {
        "question": "What is the Cisco Catalyst 9300 Series?",
        "contexts": [
            "The Cisco Catalyst 9300 Series is a stackable enterprise-class access layer switch. "
            "It supports StackWise-480 and StackWise Virtual technologies. "
            "The series offers full 802.3at PoE+, 802.3bt PoE, and UPOE support."
        ],
        "answer": "The Cisco Catalyst 9300 is a stackable enterprise access layer switch with PoE+ and UPOE support.",
        "ground_truth": "The Cisco Catalyst 9300 Series is a stackable enterprise-class access layer switch supporting StackWise-480, StackWise Virtual, and various PoE standards including 802.3at, 802.3bt, and UPOE.",
    },
    {
        "question": "What are the uplink options for the Cisco Catalyst 9500?",
        "contexts": [
            "The Cisco Catalyst 9500 Series provides 40G and 100G uplink options. "
            "It supports QSFP+ and QSFP28 modules. "
            "The 9500 is designed for high-density core and aggregation deployments."
        ],
        "answer": "The Cisco Catalyst 9500 supports 40G and 100G uplinks via QSFP+ and QSFP28 modules.",
        "ground_truth": "The Cisco Catalyst 9500 Series offers 40G and 100G uplinks using QSFP+ and QSFP28 modules, designed for core and aggregation roles.",
    },
    {
        "question": "What is Cisco DNA Center?",
        "contexts": [
            "Cisco DNA Center is a network management and command center for intent-based networking. "
            "It provides automation, assurance, and analytics for campus, branch, and WAN networks. "
            "DNA Center integrates with Cisco ISE for policy enforcement."
        ],
        "answer": "Cisco DNA Center is a centralized network management platform for automation and analytics, integrating with ISE for policy.",
        "ground_truth": "Cisco DNA Center is an intent-based networking management platform offering automation, assurance, and analytics for campus, branch, and WAN, with ISE integration for policy enforcement.",
    },
    {
        "question": "What switching capacity does the Cisco Nexus 9336C-FX2 have?",
        "contexts": [
            "The Cisco Nexus 9336C-FX2 is a 1RU switch with 36 x 40/100G QSFP28 ports. "
            "It delivers 7.2 Tbps of switching capacity and 3.6 Bpps forwarding rate. "
            "It is suitable for data center spine and leaf deployments."
        ],
        "answer": "The Nexus 9336C-FX2 has 7.2 Tbps switching capacity with 36 x 100G ports.",
        "ground_truth": "The Cisco Nexus 9336C-FX2 provides 7.2 Tbps switching capacity and 3.6 Bpps forwarding rate, with 36 x 40/100G QSFP28 ports.",
    },
    {
        "question": "What is the power consumption of the Cisco ASR 1001-X?",
        "contexts": [
            "The Cisco ASR 1001-X Router supports up to 20 Gbps of IPsec throughput. "
            "It has a maximum power consumption of 200W. "
            "The router supports dual redundant power supplies."
        ],
        "answer": "The Cisco ASR 1001-X consumes up to 200W and supports dual redundant power supplies.",
        "ground_truth": "The Cisco ASR 1001-X has a maximum power consumption of 200W and supports dual redundant power supplies.",
    },
    {
        "question": "What does Cisco Crosswork Network Automation do?",
        "contexts": [
            "Cisco Crosswork Network Automation provides closed-loop automation for service providers. "
            "It includes tools for network health insights, change automation, and service path tracing. "
            "Crosswork integrates with NSO (Network Services Orchestrator) for service lifecycle management."
        ],
        "answer": "Crosswork provides closed-loop automation, health insights, and change automation for service providers, integrating with NSO.",
        "ground_truth": "Cisco Crosswork Network Automation delivers closed-loop automation for service providers, covering health insights, change automation, service path tracing, and NSO integration.",
    },
    {
        "question": "What is the memory capacity of the Cisco Catalyst 9400 Series supervisor?",
        "contexts": [
            "The Cisco Catalyst 9400 Series Supervisor 1 Module provides 16 GB of DRAM. "
            "The Supervisor 1XL Module upgrades this to 32 GB of DRAM. "
            "Both supervisors support hot-swap redundancy."
        ],
        "answer": "The 9400 Supervisor 1 has 16 GB DRAM; the Supervisor 1XL upgrades to 32 GB. Both support hot-swap redundancy.",
        "ground_truth": "The Cisco Catalyst 9400 Supervisor 1 offers 16 GB DRAM, while the Supervisor 1XL provides 32 GB DRAM, both with hot-swap redundancy support.",
    },
    {
        "question": "What WAN technologies does the Cisco ISR 4000 Series support?",
        "contexts": [
            "The Cisco ISR 4000 Series supports T1/E1, xDSL, 4G LTE, and Metro Ethernet WAN technologies. "
            "It provides integrated WAN optimization and application visibility. "
            "The ISR 4000 also supports SD-WAN when deployed with Cisco vManage."
        ],
        "answer": "The ISR 4000 supports T1/E1, xDSL, 4G LTE, Metro Ethernet, and SD-WAN via Cisco vManage.",
        "ground_truth": "The Cisco ISR 4000 Series supports T1/E1, xDSL, 4G LTE, and Metro Ethernet WAN technologies, with integrated WAN optimization and SD-WAN capability via Cisco vManage.",
    },
    {
        "question": "What is the form factor of the Cisco UCS C240 M6?",
        "contexts": [
            "The Cisco UCS C240 M6 is a 2RU rack server. "
            "It supports up to 2 x Intel Xeon Scalable processors (3rd Gen). "
            "Maximum memory capacity is 6 TB with 32 DIMM slots."
        ],
        "answer": "The Cisco UCS C240 M6 is a 2RU rack server supporting dual 3rd Gen Intel Xeon Scalable processors and up to 6 TB RAM.",
        "ground_truth": "The Cisco UCS C240 M6 is a 2RU rack server with support for 2 x 3rd Gen Intel Xeon Scalable processors and up to 6 TB of memory across 32 DIMM slots.",
    },
    {
        "question": "What is Cisco's lead time for Catalyst 9000 Series switches in the current supply chain?",
        "contexts": [
            "As of the latest supply chain update, Cisco Catalyst 9000 Series switches have an estimated lead time of 12–16 weeks. "
            "Lead times vary by SKU and region. "
            "Customers are advised to place orders early due to global component shortages."
        ],
        "answer": "Current lead time for Catalyst 9000 Series is 12–16 weeks, varying by SKU and region.",
        "ground_truth": "The current lead time for Cisco Catalyst 9000 Series switches is 12–16 weeks depending on SKU and region, with early ordering recommended due to component shortages.",
    },
]
