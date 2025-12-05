LLM_COMPARISON_PROMPT = '''
You are given two lists of parts. Your goal is to compute soft set operations on the two lists:
1. The difference between parts1 and parts2 (items in parts1 but not in parts2).
2. The intersection (items common to both lists, considering synonyms and closely related terms).
3. The difference between parts2 and parts1 (items in parts2 but not in parts1).

**Handling Synonyms**
To handle synonyms, we need to include key information about the parts. In particular, if you believe two parts are synonyms,
you need to choose which part of the two should be the "head word" which represents the part, and indicate the synonymous parts from their corresponding lists.
Specify a pair of parts in the intersection in the format:

- HEAD: part1_or_part2 /// PART1: part1 /// PART2: part2

where part1_or_part2 is the head word (either part1 or part2), part1 is the synonym from the first list, and part2 is the synonym from the second list.

Some notes on when to include a pair of parts in the intersection:
- If two items are phrased differently but mean the same thing, put them in the intersection and choose the best head word.
  Example: If 'propeller' is in parts1 and 'a propeller' is in parts2, you would choose 'propeller' as the head word and for the pair put:
  - HEAD: propeller /// PART1: propeller /// PART2: a propeller

- If one item is a more general or specific term (hypernym/hyponym), use the more general one as the head word.
  Example: If 'propeller' is in parts1 and 'propulsion component' is in parts2, choose 'propulsion component' as the head word:
  - HEAD: propulsion component /// PART1: propeller /// PART2: propulsion component

- **Do not merge distinct parts.** If two items refer to different things, they should remain separate.
  Example: If 'propeller' is in parts1 and 'rotor' is in parts2, list them separately in the difference lists.

**Output Format**
Please format the output as bulleted lists:

OUTPUT: Difference (parts1 - parts2):
- difference1_part1
- ...

OUTPUT: Intersection (parts1 & parts2):
- HEAD: part1_or_part2 /// PART1: part1 /// PART2: part2
- ...

OUTPUT: Difference (parts2 - parts1):
- difference2_part1
- ...

If any list is empty, write "None" instead. Example when parts1 and parts2 have no common elements:

OUTPUT: Difference (parts1 - parts2):
- difference1_part1
- ...

OUTPUT: Intersection (parts1 & parts2):
None

OUTPUT: Difference (parts2 - parts1):
- difference2_part1
- ...

**Process Instructions**
- Provide the output and do not say anything else. Do not provide an explanation or other commentary.
- Ensure the output follows the format.
- Ensure that any listed parts are present in the provided part lists; do NOT make up parts that are not present.
- If one of the part lists is empty (None), treat it as an empty set and perform the soft set operations as usual.

Here are the lists of parts to compare:

List 1:
{list1}

List 2:
{list2}
'''

'''
    Problem: when asking "what visible parts does this [object] in the image have which [concept] does/does not", we want to make sure
    that if [object] has a part which [concept] could reasonably have but does not in the annotations, then we should exclude that part
    when computing the intersection/differences.

    E.g.
     parts(kitchen masher object in image): frame, hanging hook, rubber grip
     parts(dry erase board): frame, board

    In this case, we want to exclude the hanging hook from the intersection/differences because the dry erase board could reasonably
    have a hanging hook but does not in the annotations.

    Positive segmentation question:
        What visible parts does this [object] have in common with [concept]?
        Answer: parts(object) & parts(concept)

    Scenarios to avoid:
        - [object] has a part which [concept] could reasonably have but does not in the annotations.

    Negative segmentation question:
        What visible parts does this [object] have which [concept] does not?
        Answer: parts(object) \ parts(concept)

    Scenarios to avoid:
        - [object] has a part which [concept] could reasonably have but does not in the annotations.

    Difference segmetation:
        What visible parts does this [object] have in common with [concept1] which [concept2] does not have?
        Answer: (parts(object) & parts(concept1)) \ parts(concept2)

    Scenarios to avoid:
        - [object] has a part which [concept1] could reasonably have but does not in the annotations.
        - [object] has a part which [concept2] could reasonably have but does not in the annotations.
'''

LLM_EXCLUSION_PROMPT = '''
Given a list of parts and a concept name, your task is to determine which parts the concept could plausibly have.

**Instructions**
1. Understand the Concept: consider the typical structure, function, and common variations of the given concept.
2. Evaluate Each Part: for each part in the provided list, determine whether the concept could reasonably contain that part. A part is considered reasonable if:
   - It is commonly found in instances of the concept.
   - It serves a function that aligns with the concept’s typical use.
   - There are well-known examples of the concept that include this part, even if not always present.
3. Output Format:
   - Provide a list of parts that the concept could reasonably contain. If there are no parts that the concept could reasonably contain, write "None".

**Edge Cases**
- If a part is only present in rare variations of the concept, it should be excluded.
- If a part is ambiguous or multifunctional, consider its most common use cases.

**Example 1**
PARTS:
- frame
- hanging hook
- rubber grip

CONCEPT: dry erase board

OUTPUT:
- frame
- hanging hook

**Example 2**
PARTS:
- frame
- hanging hook
- rubber grip

CONCEPT: balloon

OUTPUT:
None

**Process Instructions**
- Your goal is to ensure that parts are fairly assessed based on real-world plausibility.
- Feel free to think aloud, then provide your output in the mentioned format.

Here is the list of parts and the concept:

PARTS:
{parts}

CONCEPT:
{concept}
'''