from models import BagOfWords, TFIDF, LSA, GraphRank
from metrics import calculate_rouge, calculate_bleu
import pandas as pd

# Define input text and reference summary
text = """"
The Trump administration has ordered the military to start withdrawing roughly 7,000 troops from Afghanistan in the coming months, two defense officials said Thursday, an abrupt shift in the 17-year-old war there and a decision that stunned Afghan officials, who said they had not been briefed on the plans.
President Trump made the decision to pull the troops - about half the number the United States has in Afghanistan now - at the same time he decided to pull American forces out of Syria, one official said.
The announcement came hours after Jim Mattis, the secretary of defense, said that he would resign from his position at the end of February after disagreeing with the president over his approach to policy in the Middle East.
The whirlwind of troop withdrawals and the resignation of Mr. Mattis leave a murky picture for what is next in the United States’ longest war, and they come as Afghanistan has been troubled by spasms of violence afflicting the capital, Kabul, and other important areas. 
The United States has also been conducting talks with representatives of the Taliban, in what officials have described as discussions that could lead to formal talks to end the conflict.
Senior Afghan officials and Western diplomats in Kabul woke up to the shock of the news on Friday morning, and many of them braced for chaos ahead. 
Several Afghan officials, often in the loop on security planning and decision-making, said they had received no indication in recent days that the Americans would pull troops out. 
The fear that Mr. Trump might take impulsive actions, however, often loomed in the background of discussions with the United States, they said.
They saw the abrupt decision as a further sign that voices from the ground were lacking in the debate over the war and that with Mr. Mattis’s resignation, Afghanistan had lost one of the last influential voices in Washington who channeled the reality of the conflict into the White House’s deliberations.
The president long campaigned on bringing troops home, but in 2017, at the request of Mr. Mattis, he begrudgingly pledged an additional 4,000 troops to the Afghan campaign to try to hasten an end to the conflict.
Though Pentagon officials have said the influx of forces - coupled with a more aggressive air campaign - was helping the war effort, Afghan forces continued to take nearly unsustainable levels of casualties and lose ground to the Taliban.
The renewed American effort in 2017 was the first step in ensuring Afghan forces could become more independent without a set timeline for a withdrawal. 
But with plans to quickly reduce the number of American troops in the country, it is unclear if the Afghans can hold their own against an increasingly aggressive Taliban.
Currently, American airstrikes are at levels not seen since the height of the war, when tens of thousands of American troops were spread throughout the country. 
That air support, officials say, consists mostly of propping up Afghan troops while they try to hold territory from a resurgent Taliban.
"""  

reference_summary = """
The whirlwind of troop withdrawals and the resignation of Mr. Mattis leave a murky picture for what is next in the United States’ longest war, and they come as Afghanistan has been troubled by spasms of violence afflicting the capital, Kabul, and other important areas. 
They saw the abrupt decision as a further sign that voices from the ground were lacking in the debate over the war and that with Mr. Mattis’s resignation, Afghanistan had lost one of the last influential voices in Washington who channeled the reality of the conflict into the White House’s deliberations. 
Though Pentagon officials have said the influx of forces - coupled with a more aggressive air campaign - was helping the war effort, Afghan forces continued to take nearly unsustainable levels of casualties and lose ground to the Taliban.
""" 

# Generate summaries using each model
bow_summary = BagOfWords.summarize(text)
tfidf_summary = TFIDF.summarize(text)
lsa_summary = LSA.summarize(text)
graph_summary = GraphRank.summarize(text)

# Display summaries and their evaluations
print("Bag of Words Summary:", bow_summary)
# print("Bag of Words - ROUGE:", calculate_rouge(reference_summary, bow_summary))
# print("Bag of Words - BLEU:", calculate_bleu(reference_summary, bow_summary))

# print("\nTF-IDF Summary:", tfidf_summary)
# print("TF-IDF - ROUGE:", calculate_rouge(reference_summary, tfidf_summary))
# print("TF-IDF - BLEU:", calculate_bleu(reference_summary, tfidf_summary))

# print("\nLSA Summary:", lsa_summary)
# print("LSA - ROUGE:", calculate_rouge(reference_summary, lsa_summary))
# print("LSA - BLEU:", calculate_bleu(reference_summary, lsa_summary))

# print("\nGraph-Based Ranking Summary:", graph_summary)
# print("Graph-Based Ranking - ROUGE:", calculate_rouge(reference_summary, graph_summary))
# print("Graph-Based Ranking - BLEU:", calculate_bleu(reference_summary, graph_summary))