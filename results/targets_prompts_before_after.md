## Base prompt (Climate Policy Radar's annotation guideline)

Identify the climate targets in a passage from a national law, policy or UNFCCC submission.

A target is an aim to achieve a specific outcome, that is quantifiable and is given a deadline.
Target types:
- net_zero: a commitment to balance GHG emissions with removal, effectively reducing the net emissions to zero.
- reduction: a target referring to a reduction in greenhouse gas emissions, either economy-wide or for a sector.
- other: a target that does not fit into reduction or net zero but satisfies the definition of a target, e.g. renewable energy targets.
A passage can contain several target types, or none.

## After GEPA

Identify the climate targets in a passage from a national law, policy or UNFCCC submission (passages are often noisy text fragments, e.g. extracted from PDFs, with table remnants, page numbers, or partial sentences).

Output: brief reasoning, then three boolean flags: net_zero, reduction, other. A passage can have several true flags, or none.

Definition: A target is an aim to achieve a specific outcome that is BOTH quantifiable AND has a deadline (a target year or date). Expert annotators apply this strictly. If either quantification or a deadline is missing, it does not count.

Target types:
- net_zero: an explicit commitment to balance GHG emissions with removals / reach net zero (or carbon neutrality, climate neutrality) by a date. 
- reduction: a quantified target to reduce greenhouse gas EMISSIONS (economy-wide or sector-specific, e.g. X% below baseline/BAU by year, or 'halve emissions by 2030') with a deadline. Reductions in energy consumption, trips, waste, etc. are NOT GHG emission reduction targets. Vague aims such as 'decarbonization of the sector' without a quantified figure do not count. Be conservative: if the emissions-reduction statement is vague, loosely worded, ambiguous in scope, or its quantity/deadline cannot be clearly tied to it in a fragmentary passage, mark reduction false.
- other: a quantifiable, dated target that is not net zero or GHG reduction, e.g. renewable energy share (20% by 2020, 30% by 2030), EV share of fleet, energy efficiency/consumption reduction, forest cover, nearly-zero-energy building requirements with a date, etc.

What is NOT a target: project/process descriptions, activities, measures, policies, studies, institutional development, financing mechanisms, or aspirations without a number and a date. Return all flags false for such passages.

Strategy: 1) Scan for explicit net-zero/neutrality commitments with a date. 2) Scan for quantified GHG emission reductions with a date; check that the thing being reduced is really GHG emissions (not energy use or other metrics) and that it is stated clearly with a number. 3) Scan for other quantified, dated goals. 4) For each candidate, verify both quantity and deadline; discard otherwise. Set flags independently. Do not mark reduction merely because the passage mentions 'decarbonization', 'halve emissions' loosely, or energy savings; in the worked example (Andorra-style mobility/building passage with EV shares of 20%/30% by 2030, 50%/70% by 2050, 15% energy consumption reduction, 'Halve emissions' and 'Decarbonization' of transport), experts labeled net_zero=no, reduction=no, other=yes.
