"""Local question bank for testing adaptive interview and follow-up logic.

Contains questions for core Software Developer topics (Python, OOP, Data Structures)
across easy, medium, and hard difficulties, including dedicated follow-up questions.
"""

from __future__ import annotations

from adaptive_interview.models import Difficulty, Question

DEFAULT_QUESTION_BANK: list[Question] = [
    # ---------------------------------------------------------
    # Python Questions
    # ---------------------------------------------------------
    # Easy
    Question(
        id="py_easy_01",
        text="What is the difference between a list and a tuple in Python?",
        topic="Python",
        difficulty=Difficulty.EASY.value,
        follow_up=False,
    ),
    Question(
        id="py_easy_02",
        text="Explain mutable vs immutable data types in Python with examples.",
        topic="Python",
        difficulty=Difficulty.EASY.value,
        follow_up=False,
    ),
    # Medium
    Question(
        id="py_med_01",
        text="How does Python's Global Interpreter Lock (GIL) impact CPU-bound vs I/O-bound multithreading?",
        topic="Python",
        difficulty=Difficulty.MEDIUM.value,
        follow_up=False,
    ),
    Question(
        id="py_med_02",
        text="What are Python generators, how does the yield keyword work, and what are their memory benefits?",
        topic="Python",
        difficulty=Difficulty.MEDIUM.value,
        follow_up=False,
    ),
    # Hard
    Question(
        id="py_hard_01",
        text="Explain Python's internal memory management, including reference counting, arenas, and cyclic garbage collection.",
        topic="Python",
        difficulty=Difficulty.HARD.value,
        follow_up=False,
    ),
    Question(
        id="py_hard_02",
        text="How do metaclasses work in Python, and in what scenarios would you implement __new__ vs __init__ in a metaclass?",
        topic="Python",
        difficulty=Difficulty.HARD.value,
        follow_up=False,
    ),
    # Python Follow-ups
    Question(
        id="py_fu_01",
        text="Following up on memory usage: how would you use __slots__ to reduce memory overhead in classes with millions of instances?",
        topic="Python",
        difficulty=Difficulty.MEDIUM.value,
        follow_up=True,
    ),
    Question(
        id="py_fu_02",
        text="Following up on the GIL: how would you leverage multiprocessing or asyncio to bypass GIL bottlenecks in Python?",
        topic="Python",
        difficulty=Difficulty.HARD.value,
        follow_up=True,
    ),

    # ---------------------------------------------------------
    # OOP (Object-Oriented Programming) Questions
    # ---------------------------------------------------------
    # Easy
    Question(
        id="oop_easy_01",
        text="What are the four fundamental principles of Object-Oriented Programming (OOP)?",
        topic="OOP",
        difficulty=Difficulty.EASY.value,
        follow_up=False,
    ),
    Question(
        id="oop_easy_02",
        text="What is the difference between method overloading and method overriding?",
        topic="OOP",
        difficulty=Difficulty.EASY.value,
        follow_up=False,
    ),
    # Medium
    Question(
        id="oop_med_01",
        text="Explain the Open/Closed Principle (OCP) and give a code refactoring example that adheres to it.",
        topic="OOP",
        difficulty=Difficulty.MEDIUM.value,
        follow_up=False,
    ),
    Question(
        id="oop_med_02",
        text="What is composition over inheritance, and why is composition often favored in complex systems?",
        topic="OOP",
        difficulty=Difficulty.MEDIUM.value,
        follow_up=False,
    ),
    # Hard
    Question(
        id="oop_hard_01",
        text="Explain how violating the Liskov Substitution Principle (LSP) can introduce subtle runtime bugs, and how to avoid them.",
        topic="OOP",
        difficulty=Difficulty.HARD.value,
        follow_up=False,
    ),
    Question(
        id="oop_hard_02",
        text="How would you design a thread-safe, lazy-initialized Singleton pattern while preventing reflection and serialization vulnerabilities?",
        topic="OOP",
        difficulty=Difficulty.HARD.value,
        follow_up=False,
    ),
    # OOP Follow-ups
    Question(
        id="oop_fu_01",
        text="Following up on design principles: how do Abstract Base Classes (ABCs) enforce interfaces without multiple inheritance issues?",
        topic="OOP",
        difficulty=Difficulty.MEDIUM.value,
        follow_up=True,
    ),

    # ---------------------------------------------------------
    # Data Structures Questions
    # ---------------------------------------------------------
    # Easy
    Question(
        id="ds_easy_01",
        text="What are the average and worst-case time complexities for search, insert, and delete operations in a Hash Table?",
        topic="Data Structures",
        difficulty=Difficulty.EASY.value,
        follow_up=False,
    ),
    Question(
        id="ds_easy_02",
        text="Compare arrays and singly linked lists in terms of memory layout, random access, and insertion efficiency.",
        topic="Data Structures",
        difficulty=Difficulty.EASY.value,
        follow_up=False,
    ),
    # Medium
    Question(
        id="ds_med_01",
        text="How does a balanced Binary Search Tree (like AVL or Red-Black Tree) maintain O(log n) worst-case time complexity?",
        topic="Data Structures",
        difficulty=Difficulty.MEDIUM.value,
        follow_up=False,
    ),
    Question(
        id="ds_med_02",
        text="Explain how a binary heap structure efficiently implements a Priority Queue, including swim/sink operations.",
        topic="Data Structures",
        difficulty=Difficulty.MEDIUM.value,
        follow_up=False,
    ),
    # Hard
    Question(
        id="ds_hard_01",
        text="Explain the architectural design and time/space complexity of an LRU (Least Recently Used) cache using a Hash Map and Doubly Linked List.",
        topic="Data Structures",
        difficulty=Difficulty.HARD.value,
        follow_up=False,
    ),
    Question(
        id="ds_hard_02",
        text="How does a Trie (Prefix Tree) optimize autocomplete and prefix searches compared to a standard Hash Set?",
        topic="Data Structures",
        difficulty=Difficulty.HARD.value,
        follow_up=False,
    ),
    # Data Structures Follow-ups
    Question(
        id="ds_fu_01",
        text="Following up on Hash Tables: how do open addressing (linear/quadratic probing) and separate chaining compare under high load factors?",
        topic="Data Structures",
        difficulty=Difficulty.MEDIUM.value,
        follow_up=True,
    ),
    Question(
        id="ds_fu_02",
        text="Following up on LRU Cache: how would you make an LRU cache concurrency-safe under high-throughput read/write loads?",
        topic="Data Structures",
        difficulty=Difficulty.HARD.value,
        follow_up=True,
    ),
]


def get_default_question_bank() -> list[Question]:
    """Return a fresh copy of the default question bank.

    Returns:
        A list of Question objects.
    """
    return [
        Question(
            id=q.id,
            text=q.text,
            topic=q.topic,
            difficulty=q.difficulty,
            follow_up=q.follow_up,
        )
        for q in DEFAULT_QUESTION_BANK
    ]
