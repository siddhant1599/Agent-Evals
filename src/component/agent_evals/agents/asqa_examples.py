"""ALCE's four worked examples for the ASQA prompt (D-118).

Copied byte for byte from ALCE's ``prompts/asqa_default.json`` (ALCE repository, MIT licence)
at the commit in ``ALCE_COMMIT``: each example's question, its five passages in ALCE's order,
and its cited answer. ALCE re-draws two of the four for every run (Gao et al. 2023, App. C);
the Generator does the same from the run seed (D-118).

All four questions are in ASQA's train split and none is in the dev split the evaluation sample
is drawn from (checked against the official ASQA release); ``source_id`` is each one's train
``sample_id``. Changing any text here changes results, so it requires a version bump of
``asqa_alce_vanilla``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ALCE_ASQA_EXAMPLES", "ALCE_COMMIT", "QAExample"]

#: The ALCE repository commit the examples are copied from.
ALCE_COMMIT = "246c476a4edfc564266b7346b6e29ef4861ae937"


@dataclass(frozen=True, slots=True)
class QAExample:
    """One worked example: a question, its retrieved passages and an answer that cites them.

    Attributes:
        source_id: The question's ``sample_id`` in ASQA's train split, for audit.
        question: The question, verbatim.
        passages: ``(title, text)`` per passage, in the order ALCE lists them.
        answer: The cited answer, verbatim.
    """

    source_id: str
    question: str
    passages: tuple[tuple[str, str], ...]
    answer: str

    def passage_records(self) -> tuple[dict[str, str], ...]:
        """The passages as ``title``/``text`` mappings, the shape ``format_passages`` reads."""
        return tuple({"title": title, "text": text} for title, text in self.passages)


#: D-118: ALCE's four examples, in the prompt file's order.
ALCE_ASQA_EXAMPLES: tuple[QAExample, ...] = (
    QAExample(
        source_id="-5117526355389369643",
        question="Which is the most rainy place on earth?",
        passages=(
            (
                "Cherrapunji",
                (
                    "Cherrapunji Cherrapunji (; with the native name Sohra being more commonly "
                    "used, and can also be spelled Cherrapunjee or Cherrapunji) is a "
                    "subdivisional town in the East Khasi Hills district in the Indian state of "
                    'Meghalaya. It is the traditional capital of aNongkhlaw "hima" (Khasi tribal '
                    "chieftainship constituting a petty state), both known as Sohra or Churra. "
                    "Cherrapunji has often been credited as being the wettest place on Earth, but "
                    "for now nearby Mawsynram currently holds that distinction. Cherrapunji still "
                    "holds the all-time record for the most rainfall in a calendar month for July "
                    "1861 and most rain in a year from August 1860 to July 1861, however: it "
                    "received in"
                ),
            ),
            (
                "Cherrapunji",
                (
                    "Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM "
                    "frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being "
                    "more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is "
                    "a subdivisional town in the East Khasi Hills district in the Indian state of "
                    'Meghalaya. It is the traditional capital of aNongkhlaw "hima" (Khasi tribal '
                    "chieftainship constituting a petty state), both known as Sohra or Churra. "
                    "Cherrapunji has often been credited as being the wettest place on Earth, but "
                    "for now nearby Mawsynram currently holds that distinction. Cherrapunji still "
                    "holds the all-time record for the most rainfall"
                ),
            ),
            (
                "Mawsynram",
                (
                    "Mawsynram Mawsynram () is a village in the East Khasi Hills district of "
                    "Meghalaya state in north-eastern India, 65 kilometres from Shillong. "
                    "Mawsynram receives one of the highest rainfalls in India. It is reportedly "
                    "the wettest place on Earth, with an average annual rainfall of 11,872 mm, "
                    "but that claim is disputed by Lloró, Colombia, which reported an average "
                    "yearly rainfall of 12,717 mm between 1952 and 1989 and López de Micay, also "
                    "in Colombia, which reported an annual 12,892 mm per year between 1960 and "
                    '2012. According to the "Guinness Book of World Records", Mawsynram received '
                    "of rainfall in 1985. Mawsynram is located at 25° 18′"
                ),
            ),
            (
                "Earth rainfall climatology",
                (
                    "Pacific Northwest, and the Sierra Nevada range are the wetter portions of "
                    "the nation, with average rainfall exceeding per year. The drier areas are "
                    "the Desert Southwest, Great Basin, valleys of northeast Arizona, eastern "
                    "Utah, central Wyoming, eastern Oregon and Washington and the northeast of "
                    "the Olympic Peninsula. The Big Bog on the island of Maui receives, on "
                    "average, every year, making it the wettest location in the US, and all of "
                    "Oceania. The annual average rainfall maxima across the continent lie across "
                    "the northwest from northwest Brazil into northern Peru, Colombia, and "
                    "Ecuador, then along the Atlantic coast of"
                ),
            ),
            (
                "Going to Extremes",
                (
                    "in the world. Oymyakon in Siberia, where the average winter temperature is "
                    "−47 °F (− 44 °C). Arica in Chile, where there had been fourteen consecutive "
                    "years without rain. Fog is the only local source of water. Mawsynram in "
                    "India, where average annual rainfall is 14 meters, falling within a "
                    "four-month period in the monsoon season. The rainfall is approximately equal "
                    "to that of its neighbor Cherrapunji. Dallol in Ethiopia, known as the "
                    "'Hell-hole of creation' where the temperature averages 94 °F (34 °C) over "
                    "the year. In his second series, Middleton visited places without permanent "
                    'towns, locations where "survival"'
                ),
            ),
        ),
        answer=(
            "Several places on Earth claim to be the most rainy, such as Lloró, Colombia, "
            "which reported an average annual rainfall of 12,717 mm between 1952 and "
            "1989, and López de Micay, Colombia, which reported an annual 12,892 mm "
            "between 1960 and 2012 [3]. However, the official record is held by "
            "Mawsynram, India with an average annual rainfall of 11,872 mm [3], although "
            "nearby town Sohra, India, also known as Cherrapunji, holds the record for "
            "most rain in a calendar month for July 1861 and most rain in a year from "
            "August 1860 to July 1861 [1]."
        ),
    ),
    QAExample(
        source_id="-3207884824372475669",
        question="When did the us break away from england?",
        passages=(
            (
                "United States withdrawal from Saudi Arabia",
                (
                    "United States withdrawal from Saudi Arabia Beginning during Operation Desert "
                    "Shield in August 1990, while preparing for the Gulf War, the United States "
                    "sent a large troop contingent to Saudi Arabia. After the war, remnant "
                    "troops, primarily U.S. Air Force personnel, augmented by a smaller number of "
                    "coordinating and training personnel from the U.S. Navy, U.S. Army and U.S. "
                    "Marine Corps remained in Saudi Arabia under the aegis of Joint Task Force "
                    "Southwest Asia (JTF-SWA), as part of Operation Southern Watch (OSW). The "
                    "United Kingdom and France also maintained a small contingent of Royal Air "
                    "Force and French Air Force"
                ),
            ),
            (
                "Decolonization of the Americas",
                (
                    'and France has fully "integrated" most of its former colonies as fully '
                    'constituent "departments" of France. The United States of America declared '
                    "independence from Great Britain on July 2, 1776 (although the event is now "
                    "commemorated on July 4, the date when the Declaration of Independence was "
                    "officially adopted by Congress), in so doing becoming the first independent, "
                    "foreign-recognized nation in the Americas and the first European colonial "
                    "entity to break from its mother country. Britain formally acknowledged "
                    "American independence in 1783 after its defeat in the American Revolutionary "
                    "War. Although initially occupying only the land east of the Mississippi"
                ),
            ),
            (
                "American Revolution",
                (
                    "second British army at Yorktown in the fall of 1781, effectively ending the "
                    "war. The Treaty of Paris was signed September 3, 1783, formally ending the "
                    "conflict and confirming the new nation's complete separation from the "
                    "British Empire. The United States took possession of nearly all the "
                    "territory east of the Mississippi River and south of the Great Lakes, with "
                    "the British retaining control of Canada and Spain taking Florida. Among the "
                    "significant results of the revolution was the creation of the United States "
                    "Constitution, establishing a relatively strong federal national government "
                    "that included an executive, a national judiciary, and"
                ),
            ),
            (
                "Decolonization",
                (
                    "accelerate decolonialization and bring an end to the colonial empires of its "
                    "Western allies, most importantly during the 1956 Suez Crisis, but American "
                    "military bases were established around the world and direct and indirect "
                    'interventions continued in Korea, Indochina, Latin America ("inter alia", '
                    "the 1965 occupation of the Dominican Republic), Africa, and the Middle East "
                    "to oppose Communist invasions and insurgencies. Since the dissolution of the "
                    "Soviet Union, the United States has been far less active in the Americas, "
                    "but invaded Afghanistan and Iraq following the September 11 attacks in 2001, "
                    "establishing army and air bases in Central Asia. Before"
                ),
            ),
            (
                "Decolonization",
                (
                    "the responsibility of the United Kingdom (with a copy of the new "
                    "constitution annexed), and finally, if approved, issuance of an Order of "
                    "Council fixing the exact date of independence. After World War I, several "
                    "former German and Ottoman territories in the Middle East, Africa, and the "
                    "Pacific were governed by the UK as League of Nations mandates. Some were "
                    "administered directly by the UK, and others by British dominions – Nauru and "
                    "the Territory of New Guinea by Australia, South West Africa by the Union of "
                    "South Africa, and Western Samoa by New Zealand. Egypt became independent in "
                    "1922,"
                ),
            ),
        ),
        answer=(
            "The United States took the first step towards gaining independence from "
            "Great Britain when it declared independence from Great Britain on July 2, "
            "1776 (although the event is now commemorated on July 4, 1776, the date when "
            "the Declaration of Independence was officially adopted by Congress) [2]. The "
            "Treaty of Paris was later signed on September 3, 1783, formally separating "
            "the United States from the British Empire [3]."
        ),
    ),
    QAExample(
        source_id="3219470493874132834",
        question="Who set the record for longest field goal?",
        passages=(
            (
                "Field goal",
                (
                    "toward its own end. The longest field goal kick in NFL history is 64 yards, "
                    "a record set by Matt Prater on December 8, 2013. The previous record was 63, "
                    "originally set by Tom Dempsey (1970) and then matched by Jason Elam (1998), "
                    "Sebastian Janikowski (2011), David Akers (2012), and Graham Gano (2018). "
                    "High school, college and most professional football leagues offer only a "
                    "three-point field goal; however, some professional leagues have encouraged "
                    'more rare kicks through "four-point field goals". NFL Europe encouraged long '
                    "field goals of 50 yards or more by making those worth four points instead of "
                    "three"
                ),
            ),
            (
                "Field goal range",
                (
                    "35 and 40 yard lines (closer in a crosswind) often will go for the more "
                    "risky fourth down conversion rather than risk either the touchback or the "
                    "missed field goal. The longest field goal in recorded football history was "
                    "69 yards, set by collegiate kicker Ove Johansson, who was born in Sweden, in "
                    "a 1976 Abilene Christian University football game against East Texas State "
                    "University (now Texas A&M Commerce) at Shotwell Stadium in Abilene. The "
                    "longest successful field goal in the NFL was 64 yards and was completed by "
                    "Matt Prater in 2013. The NCAA record is 67 yards held"
                ),
            ),
            (
                "Field goal",
                (
                    "both end zones) is only 66 yards. Scaccia, while playing indoor football, "
                    "attempted a 64-yard kick that was inches short of success, hitting the "
                    "crossbar. Longer field goals have been attempted at times; the longest "
                    "attempt in the NFL, which was well short and was kicked into the wind, was "
                    "76 yards, attempted by Sebastian Janikowski of the Oakland Raiders, in a "
                    "September 28, 2008 game against the San Diego Chargers. NFL Europe rewarded "
                    "kickers that successfully kicked a field goal of longer than 50 yards with a "
                    "bonus point, making such field goals worth 4 points instead of 3;"
                ),
            ),
            (
                "Field goal",
                (
                    "this accomplishment is not the official record. All of the above kicks were "
                    "successful with the use of a kicking tee, which was banned by the NCAA after "
                    "the 1988 season. The longest known drop-kicked field goal in college "
                    "football was a 62-yard kick from Pat O'Dea, an Australian kicker who played "
                    "on the Wisconsin Badgers football team. O'Dea's kick took place in a "
                    "blizzard against Northwestern on November 15, 1898. The longest field goal "
                    "in U Sports football history is 59 yards, by Niko Difonte of Calgary Dinos, "
                    "playing against the UBC Thunderbirds on November 11, 2017. The field"
                ),
            ),
            (
                "Field goal range",
                (
                    "NFL and have been banned from NCAA since 1989) is 68 yards held by Fabrizio "
                    "Scaccia, and the high school record 68 yards held by Dirk Borgognone; high "
                    "school has wider goal posts and treats a field goal attempt that lands short "
                    "in the field of play the same as a punt, making longer attempts much less "
                    "risky. The indoor football record, with narrower and higher goal posts, is "
                    "63 yards (set by Aaron Mills), which is practically as long of a field goal "
                    "as is possible in that variant of the sport, since the field in indoor "
                    "football (including"
                ),
            ),
        ),
        answer=(
            "The record for the longest field goal in an NFL game was set by Matt Prater "
            "at 64 yards [1], but the record for the longest field goal at any level was "
            "69 yards, kicked by collegiate kicker Ove Johansson in a 1976 Abilene "
            "Christian University football game against East Texas State University [2]."
        ),
    ),
    QAExample(
        source_id="-8254578338731748300",
        question="Who played galen in planet of the apes?",
        passages=(
            (
                "Planet of the Apes",
                (
                    "installment. Jacobs died on June 27, 1973, bringing an end to the APJAC "
                    'Productions era of the "Planet of the Apes" franchise. Former Fox executive '
                    'Stan Hough took over as producer for the television project, titled "Planet '
                    'of the Apes". CBS picked up the series for its 1974 autumn lineup. Ron '
                    "Harper and James Naughton played Alan Virdon and Peter Burke, two "
                    "20th-century American astronauts who pass through a time warp to a future "
                    "where apes subjugate humans (unlike the original film, the humans can "
                    "speak). Roddy McDowall returned to the franchise as Galen, a chimpanzee who "
                    "joins the astronauts."
                ),
            ),
            (
                "Planet of the Apes (1968 film)",
                (
                    "chimpanzees: animal psychologist Zira (Kim Hunter) and surgeon Galen (Wright "
                    'King). While unable to speak as his throat wound is healing, called "Bright '
                    'Eyes" by Zira and placed with one of the captive primitive humans he later '
                    'names "Nova", Taylor observes the enhanced society of talking apes and in a '
                    "strict caste system: the gorillas being the military police, hunters and "
                    "workers; the orangutans overseeing the affairs of government, science, and "
                    "religion; and intellectual chimpanzees being mostly scientists. While their "
                    "society is a theocracy similar to the beginnings of the human Industrial "
                    "Era, the apes consider the primitive humans as"
                ),
            ),
            (
                "Planet of the Apes (1968 film)",
                (
                    "Planet of the Apes (1968 film) Planet of the Apes is a 1968 American science "
                    "fiction film directed by Franklin J. Schaffner. It stars Charlton Heston, "
                    "Roddy McDowall, Kim Hunter, Maurice Evans, James Whitmore, James Daly and "
                    "Linda Harrison. The screenplay by Michael Wilson and Rod Serling was loosely "
                    'based on the 1963 French novel "La Planète des Singes" by Pierre Boulle. '
                    "Jerry Goldsmith composed the groundbreaking avant-garde score. It was the "
                    "first in a series of five films made between 1968 and 1973, all produced by "
                    "Arthur P. Jacobs and released by 20th Century Fox. The film tells the"
                ),
            ),
            (
                "Planet of the Apes",
                (
                    "Rupert Wyatt. To portray ape characters realistically, the production "
                    "avoided practical effects in favor of performance capture acting, partnering "
                    "with New Zealand visual effects company Weta Digital. Wyatt cast James "
                    "Franco as Will Rodman, while veteran performance capture actor Andy Serkis "
                    'signed on to star as Caesar. "Rise" debuted on August 5, 2011. Critics '
                    "reviewed it positively, especially praising the visual effects and Serkis's "
                    "performance. It was a major box office hit, taking in $482 million globally, "
                    "more than five times its $93 million budget. Weta's special effects earned "
                    "the film two Visual Effects Society Awards and an Oscar nomination"
                ),
            ),
            (
                "Planet of the Apes",
                (
                    "film stars Mark Wahlberg as astronaut Leo Davidson, who accidentally travels "
                    "through a wormhole to a distant planet where talking apes enslave humans. He "
                    "leads a human revolt and upends ape civilization by discovering that the "
                    "apes evolved from the normal earth primates who had accompanied his mission, "
                    "and arrived years before. Helena Bonham Carter played chimpanzee Ari, while "
                    "Tim Roth played the human-hating chimpanzee General Thade. The film received "
                    "mixed reviews; most critics believed it failed to compare to the original. "
                    "Much of the negative commentary focused on the confusing plot and twist "
                    "ending, though many reviewers praised the"
                ),
            ),
        ),
        answer=(
            "In the 1968 film Planet of the Apes, Galen was played by Wright King [2]. "
            "And in the tv series Planet of the Apes, Galen was played by Roddy McDowall "
            "[1]."
        ),
    ),
)
