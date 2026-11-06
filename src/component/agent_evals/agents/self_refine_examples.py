"""Worked examples for T1 Self-Refine's feedback and refine prompts (D-103).

Self-Refine's feedback and refine steps are few-shot prompts (Madaan et al. 2023, §3.1). Each
example pairs one article with a one-sentence draft that has factual errors, the feedback on
those errors, and a revised one-sentence summary that the article fully supports.

Every article comes from the XSum ``train`` split at revision
``7d4d486c2f8ef850b1a11aead99b894ff3dd7da9`` (``data.hf_revision`` in
``src/configs/data/tier1_xsum.yaml``), never from the frozen evaluation sample, which is drawn
from the test split. Articles are copied verbatim, including the dataset's own typos and
encoding errors. D-103 records how the three articles were chosen and who checked the examples.

Changing any example changes results, so it requires a version bump of both templates that use
them: ``xsum_self_feedback`` and ``xsum_self_refine``.

The ASQA examples (D-150) are built on three of ALCE's four ASQA worked examples (D-118), which are
referenced, not copied: the question, passages and ALCE answer come from ``ALCE_ASQA_EXAMPLES``.
The drafts, the feedback and the field-goal revision were written for this project. Changing any
of them requires a version bump of ``asqa_self_feedback`` and ``asqa_self_refine``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .asqa_examples import ALCE_ASQA_EXAMPLES, QAExample

__all__ = [
    "ASQA_SELF_REFINE_EXAMPLES",
    "EXAMPLE_SEPARATOR",
    "XSUM_SELF_REFINE_EXAMPLES",
    "FewShotExample",
    "QASelfRefineExample",
]

#: Separates the worked examples from each other and from the item under work, in both prompts.
EXAMPLE_SEPARATOR = "\n\n###\n\n"


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """One worked example: an article, a flawed draft, the feedback on it, and the revision.

    Attributes:
        source_id: The article's ID in the XSum train split, so it can be audited against the
            source and shown not to be in the evaluation sample.
        article: The train-split article, verbatim.
        draft: A one-sentence summary with one or two factual errors.
        feedback: One line per error, each starting with ``- ``, quoting the wrong words from
            ``draft`` and saying what the article states instead, or that it does not say it.
        revised: A one-sentence summary that the article fully supports.
    """

    source_id: str
    article: str
    draft: str
    feedback: str
    revised: str


#: D-103: three XSum train articles of the evaluation sample's median length, in the order the
#: prompts send them.
XSUM_SELF_REFINE_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        source_id="24880761",
        article=(
            "After a 22-hour journey to Baikonur in Kazakhstan, we had a 4am start. We were "
            "escorted into the cosmodrome, a collection of scruffy, space buildings in the "
            "desert.\n"
            "Then the hangar door opened, revealing the huge, red booster rockets looming in "
            "dark.\n"
            "The sun started to rise, and it was like a curtain had lifted, showing the "
            "incredible vast landscape of the desert steppe with the Space rocket crawling "
            "through it.\n"
            "We filmed the astronauts as they emerged in their spacesuits in the dark, holding "
            "the Olympic torch. They climbed onto the same bus that Yuri Gagarin used on the day "
            "he became the first man to launch into space.\n"
            "Then, with every cameraman looking for the perfect spot, there was a last dash to "
            "the platform overlooking the launch pad.\n"
            "The Soyuz rocket stood glistening in the sunrise.\n"
            "Fifteen minutes before launch, I had two cameras set up. One for the live position "
            "and my main camera for our correspondent.\n"
            "One minute to go\n"
            "Daniel Sandford stepped in front of the main camera and we had one last rehearsal. "
            'This "piece to camera" had to be a "one-take wonder" with perfect timing.\n'
            "Blast off. I saw a huge amount of smoke and flames, but there was no sound. We were "
            "1 km away. The rocket started to move. Boom! The sound had finally reached us, and "
            "Daniel started talking. I zoomed out to bring him into the picture.\n"
            "The rocket was moving fast out of shot so I panned up off Daniel, chasing the "
            "flaming rocket up into the blue sky. A perfect take. That was what we came for.\n"
            "In less than a minute, the rocket was just a burning speck. The torch, strapped "
            "inside the capsule on top of the rocket was heading out of the atmosphere."
        ),
        draft=(
            "A camera operator describes a 12-hour journey to Baikonur in Kazakhstan to film the "
            "Olympic torch being launched into space on a Soyuz rocket, which they watched from "
            "5 km away."
        ),
        feedback=(
            '- "a 12-hour journey": the article says the journey to Baikonur took 22 hours.\n'
            '- "5 km away": the article says they were 1 km away.'
        ),
        revised=(
            "A camera operator describes a 22-hour journey to Baikonur in Kazakhstan to film the "
            "Olympic torch being launched into space on a Soyuz rocket, which they watched from "
            "1 km away."
        ),
    ),
    FewShotExample(
        source_id="34640927",
        article=(
            "The sculpture is now part of an exhibition at the Liverpool FACT centre after it was"
            " bought at auction in conjunction with the Business Improvement District.\n"
            "It was previously in the window of a shop in Bold Street in the city centre.\n"
            "Campaigners contacted FACT and BID to save the sculpture.\n"
            "Mike Stubbs, director at FACT (the Foundation for Art and Creative Technology), said"
            ' it "made sense" to save the ship due to the importance of investing in the '
            '"heritage and local memory of Bold Street".\n'
            "\"The ship itself is an artefact with a large significance for Liverpool's cultural "
            'identity," he said.\n'
            "Mr Stubbs, who said the sculpture reminds people of strong post-war hobbyism, urged "
            "people to take the opportunity to take a closer look at it.\n"
            "Studded with red costume jewellery-type stones, a gold lion and hundreds of shells, "
            "the ship has a faded paper blue Union flag at its helm.\n"
            "In the living quarters, it contains a tiny glass swan sculpture and two floral "
            "sofas, with a Mickey Mouse figurine at the helm.\n"
            "According to the people involved in the social media campaign, the ship was worth "
            "saving and it sold for around Â£800 at auction.\n"
            "Michael Doran, communications manager at Liverpool Bid Company, said once it was "
            'sent to auction they felt "compelled" to intervene.\n'
            'He said to lose it would have been to "lose something of the eclectic spirit" which '
            "makes Bold Street so special.\n"
            "Bold Street is in the city centre and is well-known for its independent shops and "
            "restaurants.\n"
            "Curator Catherine Cox-Field said the ship reminds her of childhood trinkets that "
            "could be bought at the beach.\n"
            '"I love the fact that it was in the window of the antiques shop for so many years '
            'and that nobody knows much about it," she said.'
        ),
        draft=(
            "A ship sculpture studded with hundreds of shells, which sat for 30 years in the "
            "window of a shop on Liverpool's Bold Street, has been bought at auction and put on "
            "display at the Walker Art Gallery."
        ),
        feedback=(
            '- "for 30 years": the article does not mention how long the ship was in the window, '
            'only that it was there "for so many years".\n'
            '- "the Walker Art Gallery": the article says the ship is now part of an exhibition '
            "at the Liverpool FACT centre."
        ),
        revised=(
            "A ship sculpture studded with hundreds of shells, which sat for many years in the "
            "window of a shop on Liverpool's Bold Street, has been bought at auction and put on "
            "display at the city's FACT centre."
        ),
    ),
    FewShotExample(
        source_id="28486128",
        article=(
            "The original jewellery was found by a ploughman in the 1840s at nearby Sutton in "
            "Suffolk and what is left of it is kept at Ipswich Museum.\n"
            "Craftsmen Dave Roper has made the replica using gold, silver, bone and garnet "
            "stones.\n"
            "It ties in with the 75th anniversary of the burial ship find, in 1939.\n"
            "The ship was believed to have been the resting place of King Raedwald of East "
            "Anglia.\n"
            "The original brooch dates from the 7th Century and the National Trust, which owns "
            "the Sutton Hoo site near Woodbridge, said it had been damaged over the centuries, so"
            " that only its gold backing plate remains.\n"
            "The ploughman was believed to have taken it to a jeweller in Stowmarket and "
            "exchanged it for a set of teaspoons.\n"
            "Ruaidhri O'Mahony, visitor manager at Sutton Hoo, said: \"It's likely the brooch was "
            "plundered from one of the mounds as there was a history of grave-robbing dating back"
            " to the Tudor period.\n"
            "\"It's definitely a piece of jewellery belonging to someone of immense status and "
            "very likely was from the same workshop which created the rest of the treasures at "
            "Sutton Hoo such as the shoulder clasp and warrior's helmet.\"\n"
            "Mr Roper made the replica in his workshop in Ledbury, Herefordshire.\n"
            '"The ploughman had described the missing red stones after he sold it, but '
            'replicating the rest of it was educated guesswork based on a similar disc-brooch," '
            "he said.\n"
            '"As more research is done it illuminates the so-called Dark Ages.\n'
            '"The original craftsmen were working to a very high standard which is difficult to '
            'surpass in the modern day."\n'
            "The National Trust is recreating a 1930s garden party at Sutton Hoo this weekend to "
            "commemorate the party held by the then landowner Edith Pretty, where she displayed "
            "some of the early finds."
        ),
        draft=(
            "A craftsman in Ipswich has made a replica of a 9th-Century brooch that a ploughman "
            "found at Sutton in Suffolk in the 1840s."
        ),
        feedback=(
            '- "in Ipswich": the article says the replica was made in a workshop in Ledbury, '
            "Herefordshire.\n"
            '- "9th-Century": the article says the original brooch dates from the 7th Century.'
        ),
        revised=(
            "A craftsman in Ledbury, Herefordshire, has made a replica of a 7th-Century brooch "
            "that a ploughman found at Sutton in Suffolk in the 1840s."
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class QASelfRefineExample:
    """One ASQA worked example: an ALCE question with its passages, a flawed draft, the feedback on
    it, and the revision (D-150).

    Attributes:
        qa: ALCE's worked example, for its question, passages and ``source_id`` (D-118).
        draft: A cited answer with two errors, each a wrong citation number, a missing citation,
            or a fact that no passage states.
        feedback: One line per error, each starting with ``- ``, quoting the words from ``draft``
            and saying which search result supports them, or that none does.
        revised: A cited answer whose every sentence is supported by the passages it cites.
    """

    qa: QAExample
    draft: str
    feedback: str
    revised: str

    @property
    def source_id(self) -> str:
        """The question's ``sample_id`` in ASQA's train split, for audit."""
        return self.qa.source_id


_ALCE_BY_ID = {example.source_id: example for example in ALCE_ASQA_EXAMPLES}
_GALEN = _ALCE_BY_ID["-8254578338731748300"]
_INDEPENDENCE = _ALCE_BY_ID["-3207884824372475669"]
_FIELD_GOAL = _ALCE_BY_ID["3219470493874132834"]

#: D-150: three of ALCE's ASQA examples, in the order the prompts send them. The feedback prompt
#: shows the second as its revision with the stop word (D-103's pattern).
ASQA_SELF_REFINE_EXAMPLES: tuple[QASelfRefineExample, ...] = (
    QASelfRefineExample(
        qa=_GALEN,
        draft=(
            "In the 1968 film Planet of the Apes, Galen was played by Wright King [3]. In the 1974 "
            "television series, Galen was played by Roddy McDowall."
        ),
        feedback=(
            '- "Galen was played by Wright King [3]": search result [3] does not mention Galen or '
            "Wright King; search result [2] supports it.\n"
            '- "In the 1974 television series, Galen was played by Roddy McDowall.": this sentence '
            "cites no search result; search result [1] supports it."
        ),
        revised=_GALEN.answer,
    ),
    QASelfRefineExample(
        qa=_INDEPENDENCE,
        draft=(
            "The United States declared independence from Great Britain on July 2, 1777 [2]. The "
            "Treaty of Paris was later signed on September 3, 1783, formally separating the United "
            "States from the British Empire."
        ),
        feedback=(
            '- "July 2, 1777": no search result says this; search result [2] states that the '
            "United States declared independence on July 2, 1776.\n"
            '- "The Treaty of Paris was later signed on September 3, 1783, formally separating the '
            'United States from the British Empire.": this sentence cites no search result; search '
            "result [3] supports it."
        ),
        revised=_INDEPENDENCE.answer,
    ),
    QASelfRefineExample(
        qa=_FIELD_GOAL,
        draft=(
            "Matt Prater set the NFL record for the longest field goal, 65 yards, on December 8, "
            "2013 [1]. The longest field goal in recorded football history was 69 yards, kicked by "
            "collegiate kicker Ove Johansson in a 1976 game for Abilene Christian University "
            "against East Texas State University [4]."
        ),
        feedback=(
            '- "65 yards": no search result says this; search result [1] states that Matt '
            "Prater's record kick was 64 yards.\n"
            '- "69 yards, kicked by collegiate kicker Ove Johansson in a 1976 game for Abilene '
            'Christian University against East Texas State University [4]": search result [4] '
            "does not mention Ove Johansson; search result [2] supports it."
        ),
        revised=(
            "Matt Prater set the NFL record for the longest field goal, 64 yards, on December 8, "
            "2013 [1]. The longest field goal in recorded football history was 69 yards, kicked by "
            "collegiate kicker Ove Johansson in a 1976 game for Abilene Christian University "
            "against East Texas State University [2]."
        ),
    ),
)
