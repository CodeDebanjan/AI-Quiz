# adaptive_quiz_ui.py
"""
Adaptive AI Programming Quiz - PyQt5 single-file app

Features:
- Modern dark UI matching provided design (home/quiz/results windows)
- Upload PDF (extract text via PyPDF2) and generate customized MCQs from PDF
- Ask user how many questions to attempt
- Randomized quiz, smart repetition of wrong questions
- Explain mode (uses extracted sentence or fallback explanation)
- Result screen with embedded Matplotlib chart and CSV save
"""

import sys
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Any

import PyPDF2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette, QLinearGradient, QBrush
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QProgressBar, QMessageBox, QInputDialog, QSizePolicy,
    QStackedWidget, QFrame
)

# Matplotlib canvas for PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --------------------------
# Utility: PDF text extraction
# --------------------------
def extract_text_from_pdf(pdf_path: str, max_chars: int = 8000) -> str:
    """
    Extract text from PDF using PyPDF2.
    Return up to max_chars characters.
    """
    text_parts = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                if txt:
                    text_parts.append(txt)
                if sum(len(s) for s in text_parts) > max_chars:
                    break
    except Exception as e:
        print("PDF read error:", e)
        return ""
    return "\n".join(text_parts)[:max_chars]


# --------------------------
# Utility: Rule-based MCQ generator from text
# --------------------------
PROGRAMMING_KEYWORDS = [
    "function", "def", "class", "loop", "for", "while", "return",
    "pointer", "malloc", "printf", "scanf", "len", "list", "array",
    "object", "new", "constructor", "inheritance", "recursion", "stack",
    "queue", "pointer", "reference", "mutable", "immutable", "scope",
    "variable", "string", "int", "float", "boolean", "exception",
    "try", "catch", "import", "include"
]

# small pool of distractors to use
DISTRACTORS = [
    "function", "class", "loop", "array", "pointer", "object",
    "list", "stack", "queue", "recursion", "constructor", "variable",
    "string", "int", "float", "boolean", "mutable", "immutable", "scope"
]


def sentence_tokenize(text: str) -> List[str]:
    """
    Lightweight sentence splitter (not NLTK) — split on punctuation heuristically.
    """
    import re
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 20]
    return sents


def find_keyword_in_sentence(sentence: str) -> str:
    s = sentence.lower()
    for kw in PROGRAMMING_KEYWORDS:
        if kw in s:
            return kw
    return ""


def make_question_from_sentence(sentence: str) -> Dict[str, Any]:
    """
    Create a multiple-choice question from a sentence by masking one keyword.
    Returns dict with question, options (4), answer, explanation, 'topic'
    """
    kw = find_keyword_in_sentence(sentence)
    if not kw:
        return {}

    # question text: replace keyword with blank
    q_text = sentence.replace(kw, "______")
    # correct answer is kw (original casing not necessary)
    answer = kw

    # build distractors: pick from DISTRACTORS excluding correct
    pool = [d for d in DISTRACTORS if d.lower() != kw.lower()]
    random.shuffle(pool)
    options = [answer] + pool[:3]
    random.shuffle(options)

    explanation = sentence.strip()
    return {"question": q_text, "options": options, "answer": answer, "explanation": explanation, "topic": kw.title()}


def generate_mcqs_from_pdf_text(text: str, n_questions: int) -> List[Dict[str, Any]]:
    """
    Generate up to n_questions MCQs based on sentences that include programming keywords.
    If not enough are found, supplement with built-in fallback bank.
    """
    sents = sentence_tokenize(text)
    candidates = []
    for s in sents:
        q = make_question_from_sentence(s)
        if q:
            candidates.append(q)
    # unique by question
    unique = []
    seen = set()
    for c in candidates:
        key = c["question"]
        if key not in seen:
            unique.append(c)
            seen.add(key)
    if len(unique) >= n_questions:
        return random.sample(unique, n_questions)
    # supplement with fallback if needed
    fallback = FALLBACK_BANK.copy()
    random.shuffle(fallback)
    combined = unique + fallback
    if len(combined) >= n_questions:
        return combined[:n_questions]
    # repeat if still not enough
    while len(combined) < n_questions:
        combined.extend(random.sample(FALLBACK_BANK, min(len(FALLBACK_BANK), n_questions - len(combined))))
    return combined[:n_questions]


# --------------------------
# Fallback MCQ bank
# --------------------------
FALLBACK_BANK = [
    {"question": "Which keyword defines a function in Python?", "options": ["func", "def", "function", "lambda"], "answer": "def", "explanation": "'def' declares a function in Python.", "topic": "Python"},
    {"question": "What does len([1,2,3]) return in Python?", "options": ["1", "2", "3", "0"], "answer": "3", "explanation": "len() returns the number of elements.", "topic": "Python"},
    {"question": "Which symbol ends a statement in C?", "options": [";", ".", ":", ","], "answer": ";", "explanation": "C statements end with semicolon.", "topic": "C"},
    {"question": "Which keyword creates an object in Java?", "options": ["make", "new", "create", "instance"], "answer": "new", "explanation": "Use 'new' to instantiate objects in Java.", "topic": "Java"},
    {"question": "Which of these is mutable in Python?", "options": ["tuple", "list", "string", "int"], "answer": "list", "explanation": "Lists are mutable; tuples and strings are not.", "topic": "Python"},
    {"question": "What is recursion?", "options": ["Loop", "Function calling itself", "Variable", "Class"], "answer": "Function calling itself", "explanation": "Recursion is a function invoking itself to solve subproblems.", "topic": "Algorithms"},
    {"question": "Which operator dereferences a pointer in C?", "options": ["&", "*", "%", "#"], "answer": "*", "explanation": "'*' accesses value pointed by a pointer.", "topic": "C"},
    {"question": "Which keyword handles exceptions in Java?", "options": ["try", "catch", "throw", "handle"], "answer": "catch", "explanation": "'catch' block handles thrown exceptions.", "topic": "Java"},
]


# --------------------------
# PyQt5: Matplotlib figure widget
# --------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, width=4, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)

    def plot_bar(self, labels: List[str], values: List[float]):
        self.ax.clear()
        x = np.arange(len(labels))
        self.ax.bar(x, values)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels(labels, rotation=30, ha='right')
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel("Accuracy %")
        self.draw()


# --------------------------
# PyQt5 UI (Home / Quiz / Results)
# --------------------------
class AdaptiveQuizApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptive AI Programming Quiz")
        self.setFixedSize(900, 620)
        self.setFont(QFont("Segoe UI", 10))

        # state
        self.pdf_path = ""
        self.num_questions = 5
        self.questions: List[Dict[str, Any]] = []
        self.current_idx = 0
        self.user_answers: List[Dict[str, Any]] = []
        self.wrong_questions: List[Dict[str, Any]] = []
        self.performance: Dict[str, List[int]] = {}  # topic -> [correct, total]

        # stacked pages
        self.stack = QStackedWidget()
        self.home_page = self.make_home_page()
        self.quiz_page = self.make_quiz_page()
        self.result_page = self.make_result_page()

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.quiz_page)
        self.stack.addWidget(self.result_page)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stack)
        self.setLayout(main_layout)

        # apply stylings
        self.apply_global_style()

    # --------------------------
    # UI builders
    # --------------------------
    def make_home_page(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 30, 40, 20)
        layout.setSpacing(20)

        # Title area (center)
        title = QLabel("Adaptive AI Programming Quiz")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 26, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF;")
        layout.addSpacing(10)
        layout.addWidget(title)

        # decorative underline
        hr = QFrame()
        hr.setFixedHeight(3)
        hr.setStyleSheet("background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #00e5ff, stop:1 #00c1ff); border-radius:2px;")
        hr.setFixedWidth(420)
        hr.setFrameShape(QFrame.HLine)
        hr.setFrameShadow(QFrame.Sunken)
        hr_container = QWidget()
        hr_layout = QHBoxLayout()
        hr_layout.addStretch()
        hr_layout.addWidget(hr)
        hr_layout.addStretch()
        hr_container.setLayout(hr_layout)
        layout.addWidget(hr_container)

        # tile buttons row
        tile_row = QHBoxLayout()
        tile_row.setSpacing(28)
        tile_row.setContentsMargins(60, 20, 60, 20)

        # each tile is a rounded button-like widget
        upload_tile = self.make_tile("📑", "Upload PDF\n(optional)", self.upload_pdf_clicked)
        count_tile = self.make_tile("📖", "AI Generated Questions", self.set_num_questions_clicked)
        start_tile = self.make_tile("▶️", "Start Quiz", self.start_quiz_clicked, primary=True)

        tile_row.addStretch()
        tile_row.addWidget(upload_tile)
        tile_row.addWidget(count_tile)
        tile_row.addWidget(start_tile)
        tile_row.addStretch()

        layout.addSpacing(28)
        layout.addLayout(tile_row)

        # subtitle
        subtitle = QLabel("Click Start Quiz to begin")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet("color: #cfcfcf;")
        layout.addWidget(subtitle)

        # bottom bar area (controls)
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(10, 20, 10, 10)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_clicked)
        self.next_btn.setFixedHeight(42)
        self.next_btn.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00c9ff, stop:1 #00f0a3); color: #fff; border-radius: 20px; font-weight:600;")
        self.next_btn.setDisabled(True)  # active in quiz

        self.explain_toggle = QPushButton("Show Explanation")
        self.explain_toggle.clicked.connect(self.show_explanation_clicked)
        self.explain_toggle.setFixedHeight(42)
        self.explain_toggle.setStyleSheet("background: transparent; color: #dcdcdc; border: 1px solid rgba(255,255,255,0.06); border-radius: 20px;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(255,255,255,0.06);
                border-radius: 7px;
            }
            QProgressBar::chunk {
                border-radius:7px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00c9ff, stop:1 #00f0a3);
            }
        """)

        bottom_bar.addWidget(self.next_btn, stretch=0)
        bottom_bar.addSpacing(8)
        bottom_bar.addWidget(self.explain_toggle, stretch=0)
        bottom_bar.addSpacing(12)
        bottom_bar.addWidget(self.progress_bar, stretch=1)

        # attach
        layout.addStretch()
        layout.addLayout(bottom_bar)
        w.setLayout(layout)
        return w

    def make_tile(self, icon_text: str, label: str, callback, primary: bool = False) -> QWidget:
        tile = QWidget()
        tile_layout = QVBoxLayout()
        tile_layout.setContentsMargins(18, 18, 18, 18)
        tile_layout.setSpacing(12)

        btn = QPushButton(icon_text)
        btn.setFixedSize(110, 110)
        btn.setFont(QFont("Segoe UI Emoji", 30))
        btn.clicked.connect(callback)
        btn.setStyleSheet(self.tile_button_style(primary=primary))
        btn.setCursor(Qt.PointingHandCursor)

        lbl = QLabel(label)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #dcdcdc;")
        lbl.setFont(QFont("Segoe UI", 10))

        tile_layout.addWidget(btn, alignment=Qt.AlignCenter)
        tile_layout.addWidget(lbl)
        tile.setLayout(tile_layout)
        tile.setFixedWidth(160)
        tile.setStyleSheet("background: transparent;")
        return tile

    def tile_button_style(self, primary: bool = False) -> str:
        if primary:
            # big gradient primary tile like the image
            return """
            QPushButton {
                border-radius: 18px;
                color: white;
                font-weight: 700;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #1ec8ff, stop:1 #3be1b6);
                box-shadow: none;
            }
            QPushButton:hover {
                transform: translateY(-2px);
            }
            """
        else:
            return """
            QPushButton {
                border-radius: 18px;
                background: #ffffff;
                color: #0d3b66;
                border: none;
            }
            QPushButton:hover {
                background: #f2f2f7;
            }
            """

    def make_quiz_page(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(36, 20, 36, 16)
        layout.setSpacing(18)

        # question label
        self.question_label = QLabel("Question will appear here")
        self.question_label.setWordWrap(True)
        self.question_label.setFont(QFont("Segoe UI", 16))
        self.question_label.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(self.question_label)

        # options grid (vertical)
        self.option_buttons = []
        opts_layout = QVBoxLayout()
        opts_layout.setSpacing(12)
        for i in range(4):
            b = QPushButton("")
            b.setFixedHeight(58)
            b.setFont(QFont("Segoe UI", 12))
            b.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding-left: 16px;
                    border-radius: 12px;
                    background: rgba(255,255,255,0.06);
                    color: #e8e8e8;
                }
                QPushButton:hover { background: rgba(255,255,255,0.09); }
            """)
            b.clicked.connect(self.make_option_handler(i))
            self.option_buttons.append(b)
            opts_layout.addWidget(b)
        layout.addLayout(opts_layout)

        # right-bottom controls: progress and small hint area
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        self.question_count_label = QLabel("0 / 0")
        self.question_count_label.setStyleSheet("color: #cfcfcf;")
        self.question_count_label.setFont(QFont("Segoe UI", 10))
        bottom.addWidget(self.question_count_label)

        bottom.addStretch()
        layout.addLayout(bottom)

        # attach bottom shared controls from home (reuse self.next_btn, self.explain_toggle, self.progress_bar)
        # place them at bottom of quiz page
        layout.addStretch()
        bottom_bar = QHBoxLayout()
        bottom_bar.addWidget(self.next_btn, stretch=0)
        bottom_bar.addSpacing(8)
        bottom_bar.addWidget(self.explain_toggle, stretch=0)
        bottom_bar.addSpacing(12)
        bottom_bar.addWidget(self.progress_bar, stretch=1)
        layout.addLayout(bottom_bar)

        w.setLayout(layout)
        return w

    def make_result_page(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(36, 20, 36, 20)
        layout.setSpacing(12)

        title = QLabel("Results")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF;")
        layout.addWidget(title)

        self.score_label = QLabel("Score: 0 / 0 (0%)")
        self.score_label.setFont(QFont("Segoe UI", 14))
        self.score_label.setStyleSheet("color: #dcdcdc;")
        layout.addWidget(self.score_label)

        # chart
        self.chart = MplCanvas(width=5, height=3)
        layout.addWidget(self.chart)

        # buttons row
        row = QHBoxLayout()
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setFixedHeight(40)
        row.addWidget(self.save_btn)

        self.home_btn = QPushButton("Back to Home")
        self.home_btn.clicked.connect(self.back_to_home)
        self.home_btn.setFixedHeight(40)
        row.addWidget(self.home_btn)

        layout.addLayout(row)
        w.setLayout(layout)
        return w

    # --------------------------
    # Styling
    # --------------------------
    def apply_global_style(self):
        # background gradient similar to image
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #111316, stop:1 #1b2430);
            }
        """)
        # set window icon if available (optional)
        # self.setWindowIcon(QIcon("assets/logo.png"))

    # --------------------------
    # Button callbacks (home controls)
    # --------------------------
    def upload_pdf_clicked(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF Files (*.pdf)")
        if path:
            self.pdf_path = path
            QMessageBox.information(self, "PDF Selected", f"Selected: {Path(path).name}")

    def set_api_key_clicked(self):
        # user said they don't have api key, but we keep a placeholder to allow future key
        QMessageBox.information(self, "OpenAI Key", "You currently do not have an OpenAI API key. The app will use built-in question generation from PDF instead.")

    def set_num_questions_clicked(self):
        n, ok = QInputDialog.getInt(self, "Number of Questions", "How many coding questions do you want?", value=self.num_questions, min=1, max=50)
        if ok:
            self.num_questions = n
            QMessageBox.information(self, "Set", f"Quiz will use {self.num_questions} questions.")

    def start_quiz_clicked(self):
        # gather questions either from PDF (if provided) or fallback bank
        # generate questions then switch to quiz page
        self.questions = []
        self.user_answers = []
        self.wrong_questions = []
        self.performance = {}
        self.current_idx = 0

        if self.pdf_path and Path(self.pdf_path).exists():
            text = extract_text_from_pdf(self.pdf_path)
            if text and len(text.strip()) > 50:
                # generate
                self.questions = generate_mcqs_from_pdf_text(text, self.num_questions)
            else:
                QMessageBox.warning(self, "PDF Issue", "Could not extract usable text from PDF. Using fallback questions.")
                self.questions = random.sample(FALLBACK_BANK, self.num_questions)
        else:
            # fallback
            self.questions = random.sample(FALLBACK_BANK, self.num_questions)

        # shuffle
        random.shuffle(self.questions)
        # setup progress
        self.progress_bar.setMaximum(len(self.questions))
        self.progress_bar.setValue(0)
        self.update_question_display()
        self.next_btn.setDisabled(False)
        self.stack.setCurrentWidget(self.quiz_page)

    # --------------------------
    # Quiz mechanics
    # --------------------------
    def update_question_display(self):
        if self.current_idx >= len(self.questions):
            # done initial round
            self.handle_repetition()
            return
        q = self.questions[self.current_idx]
        self.question_label.setText(f"Q{self.current_idx+1}. {q['question']}")
        # fill options
        for i in range(4):
            opt = q["options"][i] if i < len(q["options"]) else ""
            b = self.option_buttons[i]
            b.setText(opt)
            b.setEnabled(True)
            b.setStyleSheet("""
                QPushButton {
                    text-align: left;
                    padding-left: 16px;
                    border-radius: 12px;
                    background: rgba(255,255,255,0.06);
                    color: #e8e8e8;
                }
                QPushButton:hover { background: rgba(255,255,255,0.09); }
            """)
            b.show() if opt.strip() else b.hide()
        self.question_count_label.setText(f"{self.current_idx+1} / {len(self.questions)}")
        # update progress bar
        self.progress_bar.setValue(self.current_idx)

    def make_option_handler(self, idx):
        def handler():
            if self.current_idx >= len(self.questions):
                return
            q = self.questions[self.current_idx]
            selected = self.option_buttons[idx].text().strip()
            correct = q["answer"].strip()
            is_correct = selected.lower() == correct.lower()
            self.user_answers.append({"index": self.current_idx, "selected": selected, "correct": is_correct})
            # update performance by topic
            topic = q.get("topic", "General")
            if topic not in self.performance:
                self.performance[topic] = [0, 0]
            self.performance[topic][1] += 1
            if is_correct:
                self.performance[topic][0] += 1
                # mark green briefly
                self.option_buttons[idx].setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #22c1c3, stop:1 #fdbb2d); color: #000;")
                QMessageBox.information(self, "Correct", "✅ Correct!")
            else:
                self.option_buttons[idx].setStyleSheet("background: rgba(255,80,80,0.85); color: #fff;")
                QMessageBox.information(self, "Wrong", f"❌ Wrong! Correct: {correct}")
                self.wrong_questions.append(q)
            # disable options until Next
            for btn in self.option_buttons:
                btn.setEnabled(False)
        return handler

    def next_clicked(self):
        # proceed to next question
        # re-enable next and update index
        if self.current_idx < len(self.questions):
            self.current_idx += 1
            self.update_question_display()
        else:
            # already at end, handle repetition
            self.handle_repetition()

    def show_explanation_clicked(self):
        # show explanation for current question (if exists)
        if self.current_idx < len(self.questions):
            q = self.questions[self.current_idx]
            expl = q.get("explanation", "")
            if expl:
                QMessageBox.information(self, "Explanation", expl)
            else:
                QMessageBox.information(self, "Explanation", "No explanation available for this question.")
        else:
            QMessageBox.information(self, "Explanation", "No active question.")

    def handle_repetition(self):
        # if no wrong questions, show results
        if not self.wrong_questions:
            self.show_results()
            return
        QMessageBox.information(self, "Reattempt", "You will re-attempt questions you answered incorrectly.")
        # simple reattempt loop: ask each wrong question once until correct or user cancels
        for q in list(self.wrong_questions):
            ok = False
            while not ok:
                # present as dialog asking exact option text
                opts = "\n".join([f"{i+1}. {o}" for i, o in enumerate(q["options"])])
                text, pressed = QInputDialog.getText(self, "Re-try Question", f"{q['question']}\n\n{opts}\n\nType the exact answer text (copy option):")
                if not pressed:
                    break
                if text.strip().lower() == q["answer"].strip().lower():
                    QMessageBox.information(self, "Correct", "✅ Correct!")
                    ok = True
                else:
                    QMessageBox.information(self, "Wrong", f"❌ Wrong! Correct: {q['answer']}")
                    # show explanation if available
                    if q.get("explanation"):
                        QMessageBox.information(self, "Explanation", q["explanation"])
                    # break to avoid infinite loop; user can retry again later
                    break
        # done, clear wrong questions and show results
        self.wrong_questions = []
        self.show_results()

    # --------------------------
    # Results
    # --------------------------
    def show_results(self):
        correct_count = sum(1 for a in self.user_answers if a["correct"])
        total = len(self.questions) if self.questions else 0
        pct = round((correct_count / total) * 100, 2) if total else 0
        self.score_label.setText(f"Score: {correct_count} / {total} ({pct}%)")

        # chart data
        labels = []
        values = []
        if self.performance:
            for k, v in self.performance.items():
                labels.append(k)
                acc = (v[0] / v[1]) * 100 if v[1] > 0 else 0
                values.append(round(acc, 2))
        else:
            labels = ["Overall"]
            values = [pct]
        self.chart.plot_bar(labels, values)

        # switch to result page
        self.stack.setCurrentWidget(self.result_page)

    def save_results(self):
        # save to csv
        correct_count = sum(1 for a in self.user_answers if a["correct"])
        total = len(self.questions) if self.questions else 0
        pct = round((correct_count / total) * 100, 2) if total else 0
        rec = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "questions": total, "correct": correct_count, "score_pct": pct}
        for k, v in self.performance.items():
            rec[f"perf_{k}"] = round((v[0] / v[1]) * 100 if v[1] > 0 else 0, 2)
        df = pd.DataFrame([rec])
        out = Path("quiz_results.csv")
        if out.exists():
            df.to_csv(out, mode="a", header=False, index=False)
        else:
            df.to_csv(out, index=False)
        QMessageBox.information(self, "Saved", f"Results saved to {out.resolve()}")

    def back_to_home(self):
        self.stack.setCurrentWidget(self.home_page)


# --------------------------
# Launch
# --------------------------
def main():
    app = QApplication(sys.argv)
    # set app palette slightly to improve look
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(17, 19, 22))
    palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(palette)
    # optional global stylesheet
    app.setStyleSheet("""
        QPushButton { border: none; }
        QLabel { color: #ffffff; }
    """)
    win = AdaptiveQuizApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
