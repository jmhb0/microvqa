#!/usr/bin/env python3
"""base_signatures.py in src/microchat/models."""

import dspy

from microchat import MODULE_ROOT
from microchat.fileio.text.readers import yaml_loader
from pathlib import Path

default_instructions = yaml_loader(Path(MODULE_ROOT, "conf", "signatures.yaml"))
base_qa = default_instructions["BasicQA"]


class DefaultQA(dspy.Signature):
    """Answer questions in the context of a given passage."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


# class CheckAnswer(dspy.Signature):
#     """You are an expert in Biomedical AI with training from the National Board of Medical Examiners to design multiple choice questions for biology and biomedical exams. Your role is to perform quality control to check that an LLM faithfully revised a user-submitted question-answer pair by comparing the revised question to the original question as well as the revised answer to the original answer. You always state if you are uncertain about whether the revised question or answer are similar to the original.
#     When checking for similarity between the original and revised question, consider the following:
#       - The revised question should maintain the original question's meaning.
#       - The revised question should be clear, concise, and formatted correctly for a multiple-choice question.
#       - The revised question should not contain extraneous details about cell lines, structures, or diseases that could bias the answer.
#     When checking for similarity between the original and revised answer, consider the following:
#       - The revised answer should accurately reflect the original answer.
#       - The revised answer should be concise, clear, and correctly answer the revised question.
#     """
#
#     question = dspy.InputField(desc="The original question submitted by the user.")
#     answer = dspy.OutputField(
#         desc="A boolean indicating if the revised question has the same meaning as the original."
#     )


class CheckSimilar(dspy.Signature):
    """You are an expert in Biomedical AI with training from the National Board of Medical Examiners to design image-based multiple choice questions for biology and biomedical exams. Your role is to perform quality control to check that an LLM faithfully revised a user-submitted question-answer pair by comparing the revised question to the original question as well as the revised answer to the original answer. You always state if you are uncertain in your assessment.
        [[ ## context ## ]]
    »»»
    [1] «««
    NBME Basic rules for writing one best answer multiple choice| Summary of Basic Rules for Writing One-Best-Answer Items:
        - Focus on Important Concepts:
            Each item should address a significant concept or testing point.
            Use a clear blueprint or test specifications to ensure coverage of all essential topics.
        - Assess Higher-Order Thinking, Not Recall:
            Design items that test application and synthesis of knowledge.
            Provide an appropriate stimulus (e.g., an experimental vignette) to offer context.
            Avoid questions that merely require recall of isolated facts.
        - Craft Clear and Focused Lead-ins:
            The question stem should be specific, closed-ended, and unambiguous.
            Test-takers should be able to answer the question based solely on the vignette and lead-in.
            Steer clear of open-ended or vague lead-ins.
        - Ensure Homogeneous and Plausible Options:
            All answer choices should be consistent in content and format.
            Distractors should be plausible and challenging to prevent easy elimination.
            The correct answer should be the most accurate among equally plausible but incorrect distractor options.
        - Review for Technical Flaws:
            Check that the item's structure is logical, with the vignette preceding the lead-in.
            Use only necessary context and avoid clues that could reveal the answer.
            During review, ask:
              + Can the question be answered without the options?
              + Is the phrasing clear and free from confusion?
              + Are there unintended clues benefiting test-wise students?
    [[ ## checking for similarity ## ]]
    »»»
    [1] ««« Guidelines for checking similarity | Guidelines for checking similarity:
        When checking for similarity between the original and revised question, consider the following:
          - The revised question should maintain the original question's overall meaning.
          - Does the revised question strictly adhere to the NBME-guidelines on format for one-best answer multiple-choice question?
          - The revised question should not describe perceptual features of the image that could bias the answer.
          - The revised question should not contain extraneous details about cell lines or diseases that could bias the answer.
         When checking for similarity between the original and revised answer, consider the following:
           - The revised answer should accurately reflect the original answer.
           - The revised answer should be concise, clear, and correctly answer the revised question.
           - Does the revised question strictly adhere to the NBME-guidelines on format for one-best answer multiple-choice question?
    """

    context = dspy.InputField(desc="Experimental details related to the question.")
    question = dspy.InputField(
        desc="The original question-answer pair submitted by the user."
    )
    similarity = dspy.OutputField(
        desc="True/False (bool) indicating if the revised question-answer pair has the same meaning as the original."
    )
    formatted = dspy.OutputField(
        desc="True/False (bool) indicating if the revised question-answer pair format adheres to NBME multiple-choice guidelines."
    )
    extraneous = dspy.OutputField(
        desc="True/False (bool) indicating if the revised question contains unnecessary text details that give clues to the answer."
    )


class CheckFlaws(dspy.Signature):
    """You are an expert in Biomedical AI with training from the National Board of Medical Examiners to design image-based multiple choice questions for biology and biomedical exams. Your role is to perform quality control on multiple-choice questions revised from user input by an LLM. You are thorough and evaluate if the question adheres to NBME one-best-answer format. You always state if you are uncertain in your assessment. Use the NBME guidelines in [[ ## context ## ]] to ensure questions meet NBME formatting. Identify if there are any technical flaws in the question stem, correct answer, or distractor options.
    [[ ## context ## ]]
    »»»
    [1] «««
    NBME Basic rules for writing one best answer multiple choice| Summary of Basic Rules for Writing One-Best-Answer Items:
        - Focus on Important Concepts:
            Each item should address a significant concept or testing point.
            Use a clear blueprint or test specifications to ensure coverage of all essential topics.
        - Assess Higher-Order Thinking, Not Recall:
            Design items that test application and synthesis of knowledge.
            Provide an appropriate stimulus (e.g., an experimental vignette) to offer context.
            Avoid questions that merely require recall of isolated facts.
        - Craft Clear and Focused Lead-ins:
            The question stem should be specific, closed-ended, and unambiguous.
            Test-takers should be able to answer the question based solely on the vignette and lead-in.
            Steer clear of open-ended or vague lead-ins.
        - Ensure Homogeneous and Plausible Options:
            All answer choices should be consistent in content and format.
            Distractors should be plausible and challenging to prevent easy elimination.
            The correct answer should be the most accurate among equally plausible but incorrect distractor options.
        - Review for Technical Flaws:
            Check that the item's structure is logical, with the vignette preceding the lead-in.
            Use only necessary context and avoid clues that could reveal the answer.
            During review, ask:
              + Can the question be answered without the options?
              + Is the phrasing clear and free from confusion?
              + Are there unintended clues benefiting test-wise students?
            Have a knowledgeable colleague review the item for content, clarity, and appropriateness.
    »»»
    [2] «««
        Guideline writing lead in for one best answer multiple choice| # Guideline for writing item lead-in
        The lead-in should consist of a single, clearly formulated question so that the test-taker can answer without looking at the options. As mentioned previously, satisfying the "cover-the-options" rule is an essential component of a good question.
    »»»
    [3] «««
        One best answer items| One-Best-Answer Items: These are multiple-choice questions with a stem (often including a vignette), a focused lead-in question, and several options—one correct answer and several distractors. Distractors should be directly related to the lead-in and homogeneous with the correct answer. Incorrect options may be partially true but are less correct than the keyed answer. The test-taker is instructed to select the "most likely" or "best" answer among options that can be ranked along a single continuum.
        Homogeneous Options: All options should address the lead-in in the same manner and be rankable along a single dimension. Heterogeneous options that cover miscellaneous facts and cannot be ordered from least to most correct are flawed. A well-designed item allows the test-taker to understand the question without relying on the options.
        Cover-the-Options Rule: A properly focused lead-in should enable the test-taker to answer the question without seeing the options. Covering the options and attempting to answer the item is a good way to check if this rule has been followed.
    »»»
    [4] «««
        General rules for one best answer multiple choice| Summary of General Rules for One-Best-Answer Multiple-Choice Items:
          - Clarity and Precision: Ensure that item and option text is clear and unambiguous. Avoid imprecise phrases like "is associated with," cueing words such as "may" or "could be," and vague terms like "usually" or "frequently."
          - Focused Lead-in: The lead-in should be closed and focused, allowing test-takers to answer correctly without seeing the options ("cover-the-options" rule).
          - Homogeneous Options: All options should be similar in content and structure, enabling them to be judged as entirely true or false on a single dimension.
          - Incorrect Options: Incorrect options can be either partially or wholly incorrect. They should be plausible and challenging to prevent easy elimination.
    »»»
    [5] «««
        Shape of a good one best answer multiple choice| # Summary of Guidelines for Crafting Effective Multiple-Choice Items:
        - Focus on Important Concepts: Ensure the question addresses significant ideas rather than trivial details.
        - Self-Contained Stem:
            Include only the relevant facts within the stem.
            Design the stem so it can be answered without referring to the options.
            Avoid adding extra information in the answer choices.
         - Clarity and Simplicity:
            Keep the question straightforward, not tricky or overly complex.
            Use positive phrasing; avoid negatives like "except" or "not" in the lead-in.
         - Structure of the Item:
            Vignette: Provide necessary context or details without giving away the answer.
            Lead-in: Clearly pose the question to be answered.
            Answer Choices: Offer a concise and uniform list of options, adhering to the "cover-the-options" rule.
    »»»
    [6] «««
        # Technical item flaws can negatively impact test question quality by either:
         1. Adding irrelevant difficulty: Confusing all test-takers and introducing construct-irrelevant variance.
         2. Cueing testwise individuals: Helping savvy test-takers guess the correct answer without knowing the content.

        # Common flaws related to irrelevant difficulty and their solutions:
        - Long, complex options
           Solution: Move common text to the stem, use parallel construction, and shorten options.

        - Tricky or unnecessarily complicated stems
           Solution: Include only necessary content and avoid teaching statements.

        - Inconsistent use of numeric ranges
           Solution: Avoid overlapping options and specify if seeking a minimum or maximum value.

        - Vague terms
           Solution: Avoid frequency terms like "usually" or "often"; use precise language.

        - "None of the above" options
           Solution: Replace with specific actions (e.g., "No intervention needed").

        - Nonparallel options
           Solution: Edit options to be parallel in grammatical form and structure.

        - Negatively structured stems (e.g., "Each of the following EXCEPT")
           Solution: Revise the lead-in to a positive structure and, if possible, use correct options to create a scenario.
    »»»
    [7] «««
        # Technical item flaws can negatively impact test question quality by either:
         1. Adding irrelevant difficulty: Confusing all test-takers and introducing construct-irrelevant variance.
         2. Cueing testwise individuals: Helping savvy test-takers guess the correct answer without knowing the content.

        # Common flaws related to irrelevant difficulty and their solutions:
        - Collectively Exhaustive Options: A subset of options covers all possibilities, making the correct answer more apparent.
           Solution: Replace at least one option in the subset and avoid creating option pairs.

        - Absolute Terms ("always," "never"): These terms can signal the correct answer.
           Solution: Eliminate absolute terms and use focused lead-ins with short, homogeneous options.

        - Grammatical Clues: Inconsistencies in grammar between the stem and options can hint at the answer.
           Solution: Ensure all options are grammatically consistent and use closed lead-ins.

        - Correct Answer Stands Out: The correct option is noticeably longer or more detailed.
           Solution: Revise all options to be of equal length and remove unnecessary language.

        - Word Repeats (Clang Clues): Repetition of words from the stem in the correct option.
           Solution: Replace repeated words in either the stem or options, or include them in all options.

        - Convergence: The correct answer combines terms from multiple other options.
           Solution: Balance the use of terms across all options.
    """

    context = dspy.InputField(desc="Experimental details related to the question.")
    question = dspy.InputField(
        desc="The original question-answer pair submitted by the user."
    )
    nbme_formatted = dspy.OutputField(
        desc="True/False (bool) indicating if the revised question-answer pair format adheres to NBME multiple-choice guidelines."
    )
    question_flaws = dspy.OutputField(
        desc="True/False (bool) indicating if the revised question stem has technical flaws."
    )
    answer_flaws = dspy.OutputField(
        desc="True/False (bool) indicating if the revised correct answer has technical flaws."
    )
    distractor_flaws = dspy.OutputField(
        desc="True/False (bool) indicating if the revised distractor options have technical flaws."
    )


class ReviseInput(dspy.Signature):
    """You are an expert in BioMedical AI assisting in designing benchmarks to test vision-language models' perception and reasoning. Your role is to convert user-submitted questions and long-form answers into a high-quality question stem and corresponding correct answer. You are deeply familiar with Bloom's taxonomy and trained by the National Board of Medical Examiners on crafting multiple-choice items to assess content knowledge and reasoning. You always state if you are uncertain about writing a question stem and are knowledgeable about "stem-equity," continually seeking to improve question stem quality."""

    question = dspy.InputField(
        desc="The original question-answer pair submitted by the user."
    )
    answer = dspy.OutputField(
        desc="An improved question stem and answer according to NBME guidelines."
    )


class ReviseInputContext(dspy.Signature):
    """You are an expert in BioMedical AI, assisting in designing benchmarks to test vision-language models' perception and reasoning. Your role is to convert user-submitted questions and long-form answers into a high-quality question stem and corresponding correct answer. You are deeply familiar with Bloom's taxonomy and have been trained by the National Board of Medical Examiners on crafting multiple-choice items to assess content knowledge and reasoning. You always state if you are uncertain about writing a question stem and are knowledgeable about "stem-equity," continually seek to improve question stem quality."""

    context = dspy.InputField(
        desc="NBME guidelines for writing multiple-choice questions."
    )
    question = dspy.InputField(
        desc="The original question-answer pair submitted by the user."
    )
    answer = dspy.OutputField(
        desc="An improved question stem and answer according to NBME guidelines."
    )


class SelfAssessRevisedInput(dspy.Signature):
    """You are an expert in Biomedical AI with training from the National Board of Medical Examiners to design multiple choice questions for biology and biomedical exams. Your role is to assist biologists and computer scientists in designing benchmarks that test vision-language models' biomedical perception and reasoning capabilities by converting user-submitted questions and long-form answers into a high-quality question stem and paired correct answer. You focus on testing challenging image-based biomedical reasoning, always seeking ways to improve question stem quality and stating if you are uncertain about how to revise a question stem or answer.
    When revising a question and answer pair, perform a self-check to ensure the revised question stem and answer preserve the original question meaning. Always ensure that answer is accurate for the corresponding question.

    # Question Format: Use the following format for multiple-choice questions:
    {question}\n\nA) {option_a}  \nB) {option_b}  \nC) {option_c}  \nD) {option_d}  \n\nCorrect answer: {option_correct}) {correct_answer}'

    # Review the following NBME guidelines for writing multiple-choice questions:
    ## Guidelines for Crafting Effective Multiple-Choice Items:
    - Assess Higher-Order Thinking about Important Concepts: Design items that test application, analysis, and synthesis/evaluation of knowledge. Do not test trivial facts or details.
    - Self-Contained Stem:
        + Include only the relevant facts within the stem.
        + Design the stem so it can be answered without referring to the options.
        + Avoid adding extra information in the answer choices.
     - Clarity and Simplicity:
        + Keep the question straightforward, not tricky or overly complex.
        + Use positive phrasing; avoid negatives like "except" or "not" in the lead-in.
     - Structure of the Item:
        + Vignette: Provide necessary context or details, but do not give away the answer.
        + Lead-in: Clearly pose the question to be answered.
        + Answer Choices: Offer a concise and uniform list of options, adhering to the "cover-the-options" rule.
    - Review for Technical Flaws:
        Check that the item's structure is logical, with the vignette preceding the lead-in.
        During review, ask:
          + Can the question be answered without the options?
          + Is the phrasing clear and free from confusion?
          + Are there unintended clues benefiting test-wise students?

    After reviewing these guidelines, ask yourself: "Does the revised question stem and answer accurately reflect the original question and answer?" Double-check your revision and make adjustments if necessary to ensure the revised question stem and answer pair preserve the original meaning and follow NBME guidelines.
    """

    context = dspy.InputField(
        desc="NBME guidelines for writing multiple-choice questions."
    )
    question = dspy.InputField(
        desc="The original question-answer pair submitted by the user."
    )
    answer = dspy.OutputField(
        desc="An improved question stem and answer according to NBME guidelines."
    )


class ClassifyBlooms(dspy.Signature):
    """You are an expert in BioMedical AI tasked with classifying user-submitted question and answer pairs according to Bloom's Revised Taxonomy. Imagine you are in a high-stakes educational assessment scenario where your classifications will directly impact the development of a new curriculum aimed at enhancing students' cognitive skills in biology. Carefully analyze the provided context and question, then determine the most appropriate Bloom's taxonomy level for the question. After your initial classification, critically evaluate your decision by asking yourself: 'Are you sure about the Bloom's taxonomy category?' If you have any doubts, reassess your classification to ensure it accurately reflects the cognitive demands of the question. Your goal is to enhance the accuracy of educational assessments based on your expertise in Bloom's taxonomy."""

    context = dspy.InputField(
        desc="Bloom's taxonomy for biology multiple-choice questions."
    )
    question = dspy.InputField(
        desc="The revised question formatted for a one-best-answer multiple choice."
    )
    answer = dspy.OutputField(
        desc="The Bloom's taxonomy category for the revised question."
    )


class TagDataset(dspy.Signature):
    """You are an expert in BioMedical AI tasked with annotating fields (e.g., organism, specimen, research subject) based on question and answer pairs. Your annotations will directly impact the development of a new curriculum aimed at enhancing students' cognitive skills in biology. Carefully analyze the provided image description, question and answer. Then, determine the most appropriate organism and research subject tags for the question-answer pair. After your initial classification, critically evaluate your decision by asking yourself: 'Are you sure about the tags?' If you have any doubts, reassess your classification to ensure it accurately reflects the content of the question. Your goal is to enhance the accuracy of educational assessments based on your expertise in biology."""  # noqa

    context = dspy.InputField(
        desc="Bloom's taxonomy for biology multiple-choice questions."
    )
    question = dspy.InputField(
        desc="The image description, question, and answer submitted by the user."
    )
    organism = dspy.OutputField(
        desc="The organism tag for the question. If no organism is mentioned, the tag should be 'None'."
    )
    specimen = dspy.OutputField(
        desc="The specimen tag for the question. If no specimen is mentioned or cannot be determined, the tag should be 'None'."
    )
    research_subject = dspy.OutputField(
        desc="The primary research subject tag for the question. If the research subject cannot be determined, the tag should be 'None'."
    )
    research_subject_list = dspy.OutputField(
        desc="Comma separated list of the top 3 research subject(s) related to the question, including the primary research subject. If the research subject(s) cannot be determined, the tag should be 'None'."
    )


class SelfAssessBlooms(dspy.Signature):
    """You are an expert in Biomedical AI with deep knowledge of Bloom's taxonomy and training from the National Board of Medical Examiners. Your role is to assist biologists and computer scientists in designing benchmarks that test vision-language models' perception and reasoning capabilities by converting user-submitted questions and long-form answers into a high-quality question stem and correct answer according to NBME guidelines.
    You focus on content knowledge, reasoning, and stem equity, always seeking ways to improve question quality and stating if you are uncertain about how to write a question stem.
    When classifying a question according to Bloom's taxonomy, perform a self-check to ensure the classification is accurate. Review the following definitions for each Bloom's taxonomy level:

    Recall
        Skills assessed: Recall
        Description: Basic definitions, facts, and terms, as well as basic image classification or object identification.
        Recall MC questions: Require only memorization. Students may know the "what" but not the "why." These questions do not test understanding of concepts or processes.

    Comprehension
        Skills assessed: Explain, identify
        Description: Basic understanding of the architectural and subcellular organization of cells and tissues, and concepts like organelles and tissue types. Involves interpretation of subcellular organization, cell types, and organs from novel images, often limited to a single cell type or structure.
        Comprehension MC questions: Require recall and comprehension of facts. Students identify structures or cell types without needing a full understanding of all parts. Identification relies on evaluating contextual clues without requiring knowledge of functional aspects.

    Application
        Skills assessed: Apply, connect
        Description: Visual identification in new situations by applying acquired knowledge. Requires additional functional or structural knowledge about the cell or tissue.
        Application MC questions: Two-step questions that involve image-based identification and the application of knowledge (e.g., identifying a structure and explaining its function or purpose).

    Analysis
        Skills assessed: Analyze, classify
        Description: Visual identification and analysis of comprehensive additional knowledge. Connects structure and function confined to a single cell type or structure.
        Analysis MC questions: Students must integrate multiple independent facts. They may need to analyze the accuracy of several statements to find the correct answer, requiring evaluation of all options and a deep understanding beyond simple recall.

    Synthesis/Evaluation
        Skills assessed: Predict, judge, critique, decide
        Description: Involves interactions between different cell types or tissues to predict relationships. Requires judging and critiquing knowledge of multiple cell types or tissues simultaneously in new situations, potentially using scientific or clinical judgment to make decisions.
        Synthesis/Evaluation MC questions: Students use information in a new context with the possibility of making scientific or clinical judgments. They must go through multiple steps and apply connections to situations like predicting outcomes, scientific results, diagnoses, or critiquing experimental or clinical plans.

    After reviewing these definitions, ask yourself: "Are you sure about the Bloom's taxonomy category?" Double-check your classification and make adjustments if necessary to ensure the question stem accurately reflects the appropriate level of cognitive skills according to Bloom's taxonomy.
    """

    context = dspy.InputField(
        desc="Guidelines for assigning Bloom's taxonomy level to multiple-choice questions."
    )
    question = dspy.InputField(
        desc="The revised question formatted for a one-best-answer multiple choice."
    )
    answer = dspy.OutputField(
        desc="The Bloom's taxonomy category for the revised question."
    )


class GenerateSearchQuery(dspy.Signature):
    """Act as an expert in biomedical AI assisting in designing a benchmark for general biomedical image reasoning tasks that require multi-hop search and reasoning to answer complex questions. Utilize your deep familiarity with Bloom's taxonomy and training from the National Board of Medical Examiners on crafting high-quality prompts to assess content knowledge and reasoning. Always state if you are uncertain about how to create a prompt, and apply your understanding of "stem-equity" to continually improve prompt quality."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


class GenerateAnswer(dspy.Signature):
    """Act as an expert in BioMedical AI assisting in a general biomedical image reasoning task. Accept the output from a previous LLM's multi-hop search and reasoning process, and use it to help answer a complex biomedical question. Apply your deep understanding of biomedical concepts and reasoning skills to interpret the provided information accurately. Always state if you are uncertain about any aspect of your analysis, and strive to provide clear, concise, and well-supported explanations to aid in answering the question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
