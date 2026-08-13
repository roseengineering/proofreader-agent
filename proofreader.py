#!/usr/bin/env python3

import os
import re
import sys
import json
import concurrent.futures
import urllib.parse
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

import litellm

MODEL = "openrouter/deepseek/deepseek-v4-flash"
MAX_TOKENS = 8192
MAX_TRIES = 10

SYSTEM_PROMPT = """
# You are an expert grammar writing assistant."

## Proofread the User Prompt

Proofread the entire user prompt for spelling, grammar, punctuation, and readability.  Do not alter the original tone or meaning unnecessarily.  Use the builtin 'edit' tool to perform corrections. Perform each correction one by one, and say why in the message content.  Do not perform all corrections in a single edit tool call.  Follow the rules of grammar given here strictly, don't soften them.  Do not be cautious.  Check each rule in 'Common Errors in the Use of Words' section"""

# Word Study and English Grammar 
# by Frederick W. Hamilton
# https://gutenberg.org/ebooks/30036

GRAMMAR_RULES = """
# RULES OF GRAMMAR

## The Word Families

All the words in the English language belong to one or another of nine families, each of which family has a special duty. If you will always remember to which family a word belongs and just what that family does, you will be saved from many very common errors. These nine families are: 1, nouns; 2, adjectives; 3, articles; 4, verbs; 5, pronouns; 6, adverbs; 7, prepositions; 8, conjunctions; 9, interjections. This order of enumeration is not exactly the same as will be found in the grammars. It is used here because it indicates roughly the order of the appearance of the nine families in the logical development of language. Some forms of interjections, however, may very probably have preceded any language properly so called.

## Nouns

A noun is a word used as the name of anything that can be thought of, *John*, *boy*, *paper*, *cold*, *fear*, *crowd*. There are three things about a noun which indicate its relation to other words, its number, its gender, and its case. There are two numbers, singular meaning one, and plural meaning more than one.

The plural is generally formed by adding *s* to the singular. There are a small number of nouns which form their plurals differently, *mouse*, *mice*; *child*, *children*; *foot*, *feet*. These must be learned individually from a dictionary or spelling book. There are some nouns which undergo changes in the final syllable when the *s* is added, *torch*, *torches*; *staff*, *staves*; *fly*, *flies*. These also must be learned individually. There are some nouns which have no singular, such as *cattle*, *clothes*, some which have no plural, such as *physics*, *honesty*, *news*, and some which are the same in both singular and plural, such as *deer*, *trout*, *series*. Care must be taken in the use of these nouns, as in some cases their appearance is misleading, e. g., *mathematics*, *physics*, and the like are singular nouns having no plural, but owing to their form they are often mistaken for plurals.

Compound nouns, that is to say, nouns formed by the combination of two or three words which jointly express a single idea, generally change the principal word in the forming of the plural, *hangers-on*, *ink rollers*, but in a few cases both words change, for example, *men-servants*. These forms must be learned by observation and practice. It is very important, however, that they be thoroughly learned and correctly used. Do not make such mistakes as *brother-in-laws*, *man-servants*.

Perhaps the most important use of number is in the relation between the noun and the verb. The verb as well as the noun has number forms and the number of the noun used as subject should always agree with that of the verb with which it is connected. Such expressions as "pigs is pigs," "how be you?" and the like, are among the most marked evidences of ignorance to be found in common speech. When this paragraph was originally written a group of high school boys were playing football under the writer's window. Scraps of their talk forced themselves upon his attention. Almost invariably such expressions as "you was," "they was," "he don't," "it aint," and the like took the place of the corresponding correct forms of speech.

Collective nouns, that is the nouns which indicate a considerable number of units considered as a whole, such as *herd*, *crowd*, *congress*, present some difficulties because the idea of the individuals in the collection interferes with the idea of the collection itself. The collective nouns call for the singular form of the verb except where the thought applies to the individual parts of the collection rather than to the collection as a whole, for instance, we say,

> The crowd looks large.

but we say,

> The crowd look happy.

because in one case we are thinking of the crowd and in the other of the persons who compose the crowd. So in speaking of a committee, we may say

> The Committee thinks that a certain thing should be done.

or that

> The Committee think that a certain thing should be done.

The first phrase would indicate that the committee had considered and acted on the subject and the statement represented a formal decision. The second phrase would indicate the individual opinions of the members of the committee which might be in agreement but had not been expressed in formal action. In doubtful cases it is safer to use the plural.

Entire accuracy in these cases is not altogether easy. As in the case with all the nice points of usage it requires practice and continual self-observation. By these means a sort of language sense is developed which makes the use of the right word instinctive. It is somewhat analogous to that sense which will enable an experienced bank teller to throw out a counterfeit bill instinctively when running over a large pile of currency even though he may be at some pains to prove its badness when challenged to show the reason for its rejection.

The young student should not permit himself to be discouraged by the apparent difficulty of the task of forming the habit of correct speech. It is habit and rapidly becomes easier after the first efforts.

The relation of a noun to a verb, to another noun, or to a preposition is called its case. There are three cases called the nominative, objective, and possessive. When the noun does something it is in the nominative case and is called the subject of the verb.

> The man cuts.

When the noun has something done to it it is in the objective case and is called the object of the verb.

> The man cuts paper.

When a noun depends on a preposition, it is also in the objective case and is called the object of the preposition.

> The paper is cut by machinery.

The preposition on which a noun depends is often omitted when not needed for clearness.

> The foreman gave (to) the men a holiday.\\
He came (on) Sunday.\\
Near (to) the press.\\
He was ten minutes late (late by ten minutes).\\
He is 18 years old (old by or to the extent of 18 years).

The nominative and objective cases of nouns do not differ in form. They are distinguished by their positions in the sentence and their relations to other words.

When one noun owns another the one owning is in the possessive case.

> The man's paper is cut.

The possessive case is shown by the form of the noun. It is formed by adding *s* preceded by an apostrophe to the nominative case, thus,

> John's hat.

There is a considerable difference of usage regarding the formation of the possessives of nouns ending in *s* in the singular. The general rule is to proceed as in other nouns by adding the apostrophe and the other *s* as *James's hat*. DeVinne advises following the pronunciation. Where the second *s* is not pronounced, as often happens to avoid the prolonged hissing sound of another *s*, he recommends omitting it in print.

> Moses' hat, for Moses's hat.\\
For conscience' sake.

Plural nouns ending in *s* add the apostrophe only; ending in other letters they add the apostrophe and *s* like singular nouns, *the Jones' house*, *the children's toys*.

The possessive pronouns never take the apostrophe. We say *hers*, *theirs*, *its*. *It's* is an abbreviation for *it is*.

Care should be taken in forming the possessives of phrases containing nouns in apposition, or similar compound phrases. We should say "I called at Brown the printer's" or "since William the Conqueror's time."

## Adjectives

An adjective is a word used to qualify, limit, or define a noun, or a word or phrase which has the value of a noun. Nouns are ordinarily very general and indefinite in meaning, for example, *man* conveys only a very general idea. To make that idea definite we need the help of one or more descriptive words such as *black*, *tall*, *stout*, *good*.

> I saw a man.

gives no definite idea of the person seen.

> I saw a tall, thin, dark, old man.

presents a very definite picture. It will be noted that these descriptive words have a way of forming combinations among themselves. It must be remembered, however, that all the words thus used describe the noun. Adjectives are sometimes used as substitutes for nouns. This is one of the many verbal short cuts in which the English language abounds.

> The good die young

means good people die young.

> We should seek the good and beautiful

means we should seek good or beautiful things, or persons, or qualities, or perhaps everything good and beautiful.

When adjectives indicate a quality they have three forms called degrees indicating the extent or amount of the quality possessed by the noun especially as compared with other objects of the same sort, *a big man*, *a bigger man*, *the biggest man*. These degrees are called positive, indicating possession of bigness; comparative, indicating possession of more bigness than some other man; superlative, indicating possession of more bigness than any other man. When we wish to tell the amount of the quality without comparing the possessor with any other object or group of objects we use a modifying word later to be described called an adverb.

> I saw a very big man,

indicates that the man possessed much bigness, but makes no comparison with any other man or group of men. Comparison is generally indicated in two ways, first, by adding to the adjectives the terminations *er* and *est* as *high*, *higher*, *highest*, or, second, by using the words *more* and *most*, as *splendid*, *more splendid*, *most splendid*. The question which of the two methods should be used is not always easy to decide. It depends somewhat on usage and on euphony or agreeableness of sound.

Adjectives of three or more syllables use the long form, that is, the additional word. We should not say *beautifuler* or *beautifulest*. Adjectives of two syllables may often be compared either way; for example, it would be equally correct to say *nobler* and *noblest* or *more noble* and *most noble*. An example of the influence of euphony may be found in the adjective *honest*. We might say *honester* without hesitation but we should be less likely to say *honestest* on account of the awkward combination of syllables involved. Adjectives of one syllable usually take the short form but not invariably. The exceptions, however, are more common in poetry than in prose. When any question rises it is usually safer to use the long form of comparison in the case of two-syllable adjectives and to use the short form in the case of one-syllable adjectives. The proper use of the long form is one of those niceties of diction which come only with careful observation and with training of the ear and of the literary sense.

The word *most* should never be used, as it often is, in the place of *almost*. Careless people say "I am most ready" meaning "I am almost, or nearly ready." The phrase "I am most ready," really means "I am in the greatest possible readiness." Such use of *most* is common in old English but much less so in modern speech.

Two very common adjectives are irregularly compared. They are *good*, *better*, *best*, and *bad*, *worse*, *worst*. In spite of the fact that these adjectives are among the most common in use and their comparison may be supposed to be known by everybody, one often hears the expressions *gooder*, *goodest*, *more better*, *bestest*, *bader*, *badest*, *worser*, and *worsest*. Needless to say, these expressions are without excuse except that *worser* is sometimes found in old English.

Illiterate people sometimes try to make their speech more forceful by combining the two methods of comparison in such expressions as *more prettier*, *most splendidest*. Such compounds should never be used.

Some adjectives are not compared. They are easily identified by their meaning. They indicate some quality which is of such a nature that it must be possessed fully or not at all, *yearly*, *double*, *all*. Some adjectives have a precise meaning in which they cannot be compared and a loose or popular one in which they can be; for example, a thing either is or is not *round* or *square*. Nevertheless we use these words in such a loose general way that it is not absolutely incorrect to say *rounder* and *roundest* or *squarer* and *squarest*. Such expressions should be used with great care and avoided as far as possible. None but the very ignorant would say *onliest*, but one often sees the expressions *more* and *most unique*. This is particularly bad English. Unique does not mean *rare*, *unusual*; it means one of a kind, absolutely unlike anything else. Clearly this is a quality which cannot be possessed in degrees. An object either does or does not have it.

## Articles

An article is a little adjective which individualizes the noun, *a* boy, *an* apple, *the* crowd.

*A* which is used before consonantal sounds and *an* which is used before vowel sounds are called indefinite articles because they individualize without specializing. *The* is called the definite article because it both individualizes and specializes.

*A* may be used before *o* and *u* if the sound is really consonantal as in *such a one*, *a use*, *a utility*. *An* may be used before *h* if the *h* is not sounded, for example, *an hour* but *a horror*.

## Verbs

A verb is a word which asserts or declares. In other words, it makes a noun or pronoun tell something. *John paper* tells nothing. *John wastes paper* tells something. Verbs are the most difficult of all the parts of speech to understand and to use properly. As a rule, an English verb has something more than fifty parts which, with their uses, should be thoroughly learned from a grammar. This is not so difficult a matter as it might appear, except to those whose native speech is not English. Nevertheless you should be on the guard against such blunders as *I seen*, *I seed*, for *I saw*, *I runned* for *I ran*, *I et* for *I ate*, *I throwed* for *I threw*, and the like. In most verbs these parts are regular. In some they are irregular. A list of irregular verbs will be found at the end of this volume.

While the plan of this book does not call for a systematic study of verbs any more than of any other words, it is desirable to call attention to some points as being the occasions of frequent mistakes.

A simple sentence consists of a verb, its subject, and its object. The verb indicates the action, the subject is the noun (name of a person or thing) which does the act, the object is the noun to which the thing is done. Verbs have forms denoting person and number.

  ----------------------------- ----------------------- -----------------------
            Singular                                            Plural

           1st I love                                         1st We love

   2nd You love (thou lovest)                                 2nd You love
       formal and archaic.

          3rd He loves                                       3rd They love





            Singular                                            Plural

            1st I was                                         1st We were

    2nd You were (thou wast)                                 2nd You were

           3rd He was                                        3rd They were
  ----------------------------- ----------------------- -----------------------

Verbs agree with their subjects in person and number. We all know this but we do not always remember it. Unless you are very careful, you will find yourself using a singular subject with a plural verb or the reverse. Mistakes of this sort are particularly liable to happen in the case of collective nouns, in the use of personal pronouns as subjects, and in cases where the subject and the verb are far separated in the sentence.

Those forms of the verb which tell whether the subject is acting or is acted upon are called voices. When the subject is acting the verb is said to be in the active voice. When the subject is acted upon the verb is said to be in the passive voice. Verbs in the passive voice have no objects because the subject, being acted upon, is itself in the place of an object.

Those forms of the verb which tell whether the time of the action is past, present, or future, are called tenses. They are six, viz.

> Present, I *print* (*am printing*) the book.\\
Past or imperfect, I *printed* the book.\\
Future, I *shall print* the book.\\
Perfect, or present perfect, I *have printed* the book.\\
Pluperfect or past perfect, I *had printed* the book before you wrote.\\
Future perfect, I will notify you when I *shall have printed* the book.

When adverbs denoting time are indicated care should be taken to see that the verb is consistent with the adverb. "I *printed* it yesterday," not "I *have printed* it yesterday;" "I *have not* yet *printed* it," not "I *did* not *print* it yet;" "I *have printed* it already," not "I *printed* it already."

Trouble is sometimes found in choosing the right forms of the verb to be used in subordinate clauses. The rule is:

Verbs in subordinate sentences and clauses must be governed by the tense of the principal verb.

This rule rests on the exact meaning of the forms and words used and its application can be checked by careful examination of these meanings. "He *said* he *did* it." "He *said* he *would do* it." "He *says* he *will* do it."

Note that when the statement in the subordinate clause is of universal application the present tense is always used whatever the tense of the principal verb. "The lecturer said that warm weather always softens rollers."

Those forms of the verb which tell whether the action is an actual fact, a possibility, a condition, or a command are called moods.

There are three moods, the indicative, subjunctive, and imperative.

The indicative mood indicates that the action is a fact. It is also used in asking questions.

The subjunctive mood is less used in modern than in old English. It is most commonly found in clauses beginning with *if*, though *if* is not to be regarded as the sign of the subjunctive in any such sense as *to* is the sign of the infinitive.

The subjunctive *were* should be used in purely hypothetical clauses such as "If I were in your place."

The subjunctive *be* should be used in the hypothesis or supposition of a scientific demonstration,

> If the triangle A be placed on the triangle B.

The subjunctive without *if* is often used in wishes or prayers,

> God forgive him.\\
O, that my brother were here.

The subjunctive is sometimes used to express condition,

> Had you not been a coward, you would not have run away.

The imperative mood indicates a command,

> Put that on the press.

The subject of the imperative mood is only expressed when it is emphatic,

> Go thou and do likewise.

Older grammarians speak of a fourth mood called potential. The present tendency among grammarians is to treat these forms separately. They are verb phrases which express ability, possibility, obligation, or necessity. They are formed by the use of the auxiliary verbs *may*, *can*, *must*, *might*, *could*, *would*, and *should*, with the infinitive without *to*.

*May* is used (a) to show that the subject is permitted to do something, "You may go out," or (b) to indicate possibility or doubtful intention, "I may not go to work tomorrow."

*Can* is used to show that the subject is able to do something, "I can feed a press." These two forms are often confused, with results which would be ridiculous if they were not too common to attract attention. The confusion perhaps arises from the fact that the ability to do a thing often appears to depend on permission to do it. "May I see a proof?" means "Have I permission, or will you allow me, to see a proof?" and is the proper way to put the question. The common question, "Can I see a proof?" is absurd. Of course you can, if you have normal eyesight.

*Must* shows necessity or obligation.

> You must obey the rules of the office.

*Ought* which is sometimes confounded with *must* in phrases of this sort expresses moral obligation as distinguished from necessity.

> You ought to obey the rules of the office,

indicates that it is your duty to obey because it is the right thing to do even though no penalty is attached.

> You must obey the rules of the office,

indicates that you will be punished if you do not obey.

Those forms of the verb which express the time of the action are called tenses. No particular difficulty attends the use of the tenses except in the case of *shall* and *will* and *should* and *would*.

*Shall* and *will* are used as follows: In simple statements to express mere futurity, use *shall* in the first person, *will* in the second and third; to express volition, promise, purpose, determination, or action which the speaker means to control use *will* in the first person, *shall* in the second and third.

The following tables should be learned and practiced in a large variety of combinations.

|          |           |     |                |            |
|:--------:|:---------:|-----|:--------------:|:----------:|
| Futurity |           |     | Volition, etc. |            |
| I shall  | We shall  |     |     I will     |  We will   |
| You will | You will  |     |   You shall    | You shall  |
| He will  | They will |     |    He shall    | They shall |

A good example of the misuse of the words is found in the old story of the foreigner who fell into the water and cried out in terror and despair "I *will* drown, nobody *shall* help me."

In asking questions, for the first person always use *shall*, for the second and third use the auxiliary expected in the answer.

|                     |     |                       |
|:-------------------:|:---:|:---------------------:|
|      Futurity       |     |                       |
|  Shall I (I shall)  |     |  Shall we (We shall)  |
| Shall you (I shall) |     | Shall you (We shall)  |
|  Will he (He will)  |     | Will they (They will) |
|                     |     |              |
|                     |     |                       |
|   Volition, etc.    |     |                       |
|      ---- ---       |     |       ---- ---        |
|  Will you (I will)  |     |  Will you (We will)   |
| Shall he (He shall) |     |  Shall he (He shall)  |

In all other cases, as in subordinate clauses *shall* is used in all persons to express mere futurity, *will* to express volition, etc.

In indirect discourse, when the subject of the principal clause is different from the noun clause, the usage is like that in direct statement, for example,

> The teacher says that James will win the medal. (futurity),

but when the subject of the principal clause is the same as that of the noun clause, the usage is like that in subordinate clauses,

> The teacher says that he shall soon resign. (futurity).

Exceptions. *Will* is often used in the second person to express an official command.

> You will report to the superintendent at once.

*Shall* is sometimes used in the second and third persons in a prophetic sense.

> Ye shall know the truth and the truth shall make you free.

The use of *should* and *would* is in general the same as that of *shall* and *will* in indirect statement.

|           |     |             |
|:---------:|:---:|:-----------:|
| Futurity  |     |             |
| I should  |     |  We would   |
| You would |     | You should  |
| He would  |     | They should |

In asking questions use *should* in the first person to express mere futurity and *would* to express volition, etc; in the second and third persons use the form that is expected in the answer.

|                |             |     |             |               |
|:--------------:|:-----------:|:---:|:-----------:|:-------------:|
|    Futurity    |             |     |             |               |
|    Should I    | (I should)  |     |  Should we  |  (We should)  |
|   Should You   | (I should)  |     | Should You  |  (We should)  |
|    Would he    | (He would)  |     | Would they  | (They would)  |
|                |             |     |             |      |
|                |             |     |             |               |
| Volition, etc. |             |     |             |               |
|    Would I     |  (I would)  |     |  Would we   |  (We would)   |
|   Would You    | (You would) |     |  Would You  |  (We would)   |
|   Should he    | (He should) |     | Should they | (They should) |

In subordinate clauses *should* is used in all persons to express futurity, *would* to express volition, etc.

In indirect discourse the usage is similar to that in direct statement.

> The teacher said that John would win the medal.

Exceptions. *Should* is often used to express moral obligation.

> You should be honest under all conditions.

*Would* is sometimes used to express frequentive action.

> He would walk the floor night after night.

Mistakes are often made in the use of compound tenses on account of failure to grasp the meaning of the words used.

> I should have liked to have seen you,

is correct grammar but probably not correct statement of fact, as it states a past desire to have done something at a period still further remote, that is to say, "I should have liked (yesterday) to have seen you (day before yesterday)." What is generally meant is either "I should have liked to see you," that is "I (then) wished to see you," or "I should like to have seen you," that is "I (now) wish I had seen you (then)."

Every word has its own value and nearly all our mistakes arise from lack of regard for the exact value of the words to be used.

Where a participial construction is used as the object of a verb, the noun or pronoun in the object should be in the possessive case and not in the objective. You should not say, "I object to him watching me," but "I object to his watching me."

Care should be taken not to give objects to passive verbs. The very common expression "The man was given a chance" is incorrect. It should be "A chance was given to the man."

Care should also be taken to avoid the omission of the prepositions which are needed with certain verbs, for example, "beware the dog," "What happened him" should be "beware *of* the dog," "What happened *to* him."

On the other hand superfluous prepositions are sometimes used in such phrases as *consider of*, *accept of* and the like.

Such errors are to be avoided by careful study of the meaning of words and careful observation of the best written and spoken speech.

## Pronouns

Pronouns are substitutes for nouns. They are labor saving devices. We could say everything which we need to say without them, but at the expense of much repetition of longer words. A child often says "John wants Henry's ball" instead of "I want your ball." Constant remembrance of this simple fact, that a pronoun is only a substitute for a noun, is really about all that is needed to secure correct usage after the pronouns themselves have once become familiar. A construction which appears doubtful can often be decided by substituting nouns for pronouns and vice versa.

A very common error is the use of the plural possessive pronouns with the words *any*, *every*, *each*, *somebody*, *everybody*, and *nobody*, all of which are always singular.

> We could accomplish this if every one would do their part.

is wrong. It should be

> We could accomplish this if every one would do his part.

Another common mistake is the confusion of the nominative and objective cases in objective clauses where two pronouns or a noun and a pronoun occur.

> All this was done for you and I.

is a very common but entirely inexcusable mistake. One would hardly think of saying

> "All this was done for I."\\
I saw John and he leaving the shop.

is almost equally common and quite equally bad. Do not allow yourself to be confused by a double object.

In general great care should be taken to avoid ambiguity in the use of pronouns. It is very easy to multiply and combine pronouns in such a way that while grammatical rules may not be broken the reader may be left hopelessly confused. Such ambiguous sentences should be cleared up, either by a rearrangement of the words or by substitution of nouns for some of the pronouns.

## Adverbs

An adverb is a helper to a verb, "I fear greatly," "that press works badly." Adverbs modify or help verbs, adjectives, and other adverbs just as adjectives modify nouns and pronouns. The use of adverbs presents some difficulties, mainly arising from the adverbial use of many other parts of speech and from the close relation between adverbs and adjectives.

It should never be forgotten that while adverbs never modify nouns or pronouns, adjectives never modify anything but nouns or pronouns. Remembrance of this simple fact will settle most questions as to the use of adverbs or adjectives. Careful observation and care in forming correct habits of expression will do the rest.

Do not multiply negatives. They cancel each other like the factors in an arithmetical problem. "He never did wrong" is correct in statement and clear in meaning. "He never did nothing wrong" does not add force, it reverses the meaning. The negatives have cancelled each other and you are saying "He did wrong." "He never did nothing wrong to nobody" leaves us with an odd negative and brings us back to the first statement, very badly expressed.

## Prepositions

A preposition is a hook for a noun or pronoun to hang on. It usually precedes the noun or pronoun which hangs, or depends upon it, as indicated by its name which is derived from the Latin *pre*-before and *pono*-I place.

> John is behind the press.\\
I shall work until Sunday.

A preposition shows the relation of a noun or pronoun used as its object to some other word or words in the sentence or, as it has been otherwise stated, makes the noun or pronoun to which it is joined equivalent to an adjective or an adverb. The expression "John is behind the press" is equivalent to an adjective describing John. That is, he is "John behind-the-press." Prepositions are governing words and the words governed by or depending on them are always in the objective case.

## Conjunctions

A conjunction is the coupling link between the parts of a train of thought. It is of no purpose whatever except to connect.

> I am cold and hungry and tired and I am going home.

Care should be taken to avoid confusing *and* and *but* and *and* and *or*.

> He sees the right and does the wrong.

should be

> He sees the right but does the wrong.

The ideas are contrasted, not associated.

> I did not see Thomas and John.

should be

> I did not see Thomas or John.

The first phrase means that I did not see them together, it says nothing about seeing them separately.

*Either*—*or* and *neither*—*nor* are called correlative conjunctions. They should always be paired in this way. *Neither* should never be paired with *or* nor *either* with *nor*. Each member of the pair should be placed in the same relative position, that is before the same part of speech.

> I could neither see him nor his father.

is wrong. It should be

> I could see neither him nor his father.

This rule applies to all other correlatives, that is since they are correlatives in form they should be correlatives in position also. It is correct to say

> It belongs both to you and to me.

or

> It belongs to both you and me.

but not

> It belongs both to you and me.

## Interjections

An interjection is a word or sound expressing emotion only such as a shout, a groan, a hiss, a sob, or the like, such as *Oh*, *alas*, *hush*.

## General Notes

The position of words in a sentence is often very important. Misplacement will frequently cause ambiguities and absurdities which punctuation will not remove. What does the phrase "I only saw him" mean? A newspaper advertisement describing a certain dog which was offered for sale says "He is thoroughly house-broken, will eat anything, is very fond of children." As a rule modifiers should be kept close to the words, clauses, or phrases which they modify, but due regard should be given to sense and to ease of expression.

A word or phrase which can be easily supplied from the context may often be omitted. Care must be used in making these omissions or the result will be either ambiguous or slovenly.

> Washington is nearer New York than Chicago.

What exactly does this mean? One might get into serious trouble over the interpretation of the phrase "He likes me better than you."

*All day* and *all night* are recognized as good expressions sanctioned by long usage. *All morning* and *all afternoon* are not yet sanctioned by good usage and give a decided impression of slovenliness.

Another objectionable omission is that of *to* before *place* and similar words in such expressions as "Let's go some place" and the like. It should be *to some place* or, generally better, *somewhere*.

A decidedly offensive abbreviation is the phrase *Rev. Smith*. It should be *Rev. John Smith* or *Rev. Mr. Smith*. *Rev.* is not a title, or a noun in apposition, but an adjective. It would be entirely correct to say *Pastor Smith* or *Bishop Smith*. The same error sometimes occurs in using the prefix *Hon.*

A knowledge of the correct use and combination of words is fully as important as a knowledge of their grammatical forms and their relations. This knowledge should be acquired by the use of books on rhetoric and by careful study of words themselves. The materials for such study may be found in the books named in the "Supplementary Reading" or in other books of a similar character.

The task of the writer or speaker is to say what he has to say correctly, clearly, and simply. He must say just what he means. He must say it definitely and distinctly. He must say it, so far as the subject matter will permit, in words that people of ordinary intelligence and ordinary education cannot misunderstand. "The right word in the right place" should be the motto of every man who speaks or writes, and this rule should apply to his everyday talk as well as to more formal utterances.

Three abuses are to be avoided.

Do not use slang as a means of expression. There are occasions when a slang phrase may light up what you are saying or may carry it home to intellects of a certain type. Use it sparingly if at all, as you would use cayenne pepper or tabasco sauce. Do not use it in writing at all. Slang is the counterfeit coin of speech. It is a substitute, and a very poor substitute, for language. It is the refuge of those who neither understand real language nor know how to express themselves in it.

Do not use long, unusual words. Use short and simple words whenever they will serve your turn. It is a mistake to suppose that a fluent use of long words is a mark either of depth of thought or of extent of information. The following bit of nonsense is taken from the news columns of a newspaper of good standing: "The topography about Puebla avails itself easily to a force which can utilize the heights above the city with cannon." What was meant was probably something like this, "The situation of Puebla is such as to give a great advantage to a force which can plant cannon on the high ground overlooking the city."

Do not use inflated or exaggerated words.

A *heavy shower* is not a *cloud burst*; a *gale* is not a *blizzard*; a *fire* is not a *conflagration*; an *accident* or a *defeat* is not a *disaster*; a *fatal accident* is not a *holocaust*; a *sharp criticism* is not an *excoriation* or *flaying*, and so on.

## Rules for Correct Writing

More than a century ago the great Scotch rhetorician Campbell framed five canons or rules for correct writing. They have never been improved. They should be learned by heart, thoroughly mastered, and constantly practiced by every writer and speaker. They are as follows:

Canon 1.—When, of two words or phrases in equally good use, one is susceptible of two significations and the other of but one, preference should be given to the latter: e. g., *admittance* is better than *admission*, as the latter word also means *confession*; *relative* is to be preferred to *relation*, as the latter also means the telling of a story.

Canon 2.—In doubtful cases regard should be given to the analogy of the language; *might better* should be preferred to *had better*, and *would rather* is better than *had rather*.

Canon 3.—The simpler and briefer form should be preferred, other things being equal, e. g., omit the bracketed words in expressions such as, *open* (*up*), *meet* (*together*), *follow* (*after*), *examine* (*into*), *trace* (*out*), *bridge* (*over*), *crave* (*for*), etc.

Canon 4.—Between two forms of expression in equally good use, prefer the one which is more euphonious: e. g., *most beautiful* is better than *beautifullest*, and *more free* is to be preferred to *freer*.

Canon 5.—In cases not covered by the four preceding canons, prefer that which conforms to the older usage: e. g., *begin* is better than *commence*.

## The Sentence

The proper construction of sentences is very important to good writing. The following simple rules will be of great assistance in sentence formation. They should be carefully learned and the pupil should be drilled in them.

1. Let each sentence have one, and only one, principal subject of thought. Avoid heterogeneous sentences.

2. The connection between different sentences must be kept up by adverbs used as conjunctions, or by means of some other connecting words at the beginning of the sentence.

3. The connection between two long sentences or paragraphs sometimes requires a short intervening sentence showing the transition of thought.

## The Paragraph

The proper construction of paragraphs is also of great importance. The following rules will serve as guides for paragraphing. They should be learned and the pupil should be drilled in their application.

1. A sentence which continues the topic of the sentence which precedes it rather than introduces a new topic should never begin a paragraph.

2. Each paragraph should possess a single central topic to which all the statements in the paragraph should relate. The introduction of a single statement not so related to the central topic violates the unity.

3. A sentence or short passage may be detached from the paragraph to which it properly belongs if the writer wishes particularly to emphasize it.

4. For ease in reading, a passage which exceeds three hundred words in length may be broken into two paragraphs, even though no new topic has been developed.

5. Any digression from the central topic, or any change in the viewpoint in considering the central topic, demands a new paragraph.

6. Coherence in a paragraph requires a natural and logical order of development.

7. Smoothness of diction in a paragraph calls for the intelligent use of proper connective words between closely related sentences. A common fault, however, is the incorrect use of such words as *and* or *but* between sentences which are not closely related.

8. In developing the paragraph, emphasis is secured by a careful consideration of the relative values of the ideas expressed, giving to each idea space proportionate to its importance to the whole. This secures the proper climax.

9. The paragraph, like the composition itself, should possess clearness, unity, coherence, and emphasis. It is a group of related sentences developing a central topic. Its length depends upon the length of the composition and upon the number of topics to be discussed.

## Rules for the Use and Arrangement of Words

The following rules for the use and arrangement of words will be found helpful in securing clearness and force.

1. Use words in their proper sense.

2. Avoid useless circumlocution and "fine writing."

3. Avoid exaggerations.

4. Be careful in the use of *not* ... *and*, *any*, *but*, *only*, *not* ... *or*, *that*.

5. Be careful in the use of ambiguous words, e. g., *certain*.

6. Be careful in the use of *he*, *it*, *they*, *these*, etc.

7. Report a speech in the first person where necessary to avoid ambiguity.

8. Use the third person where the exact words of the speaker are not intended to be given.

9. When you use a participle implying *when*, *while*, *though*, or *that*, show clearly by the context what is implied.

10. When using the relative pronoun, use *who* or *which*, if the meaning is *and he* or *and it*, *for he* or *for it*.

11. Do not use *and which* for *which*.

12. Repeat the antecedent before the relative where the non-repetition causes any ambiguity.

13. Use particular for general terms. Avoid abstract nouns.

14. Avoid verbal nouns where verbs can be used.

15. Use particular persons instead of a class.

16. Do not confuse metaphor.

17. Do not mix metaphor with literal statement.

18. Do not use poetic metaphor to illustrate a prosaic subject.

19. Emphatic words must stand in emphatic positions; i. e., for the most part, at the beginning or the end of the sentence.

20. Unemphatic words must, as a rule, be kept from the end.

21. The Subject, if unusually emphatic, should often be transferred from the beginning of the sentence.

22. The object is sometimes placed before the verb for emphasis.

23. Where several words are emphatic make it clear which is the most emphatic. Emphasis can sometimes be given by adding an epithet, or an intensifying word.

24. Words should be as near as possible to the words with which they are grammatically connected.

25. Adverbs should be placed next to the words they are intended to qualify.

26. *Only*; the strict rule is that *only* should be placed before the word it affects.

27. When *not only* precedes *but also* see that each is followed by the same part of speech.

28. *At least*, *always*, and other adverbial adjuncts sometimes produce ambiguity.

29. Nouns should be placed near the nouns that they define.

30. Pronouns should follow the nouns to which they refer without the intervention of any other noun.

31. Clauses that are grammatically connected should be kept as close together as possible. Avoid parentheses.

32. In conditional sentences the antecedent or "if-clauses" must be kept distinct from the consequent clauses.

33. Dependent clauses preceded by *that* should be kept distinct from those that are independent.

34. Where there are several infinitives those that are dependent on the same word must be kept distinct from those that are not.

35. In a sentence with *if*, *when*, *though*, etc. put the "if-clause" first.

36. Repeat the subject where its omission would cause obscurity or ambiguity.

37. Repeat a preposition after an intervening conjunction especially if a verb and an object also intervene.

38. Repeat conjunctions, auxiliary verbs, and pronominal adjectives.

39. Repeat verbs after the conjunctions *than*, *as*, etc.

40. Repeat the subject, or some other emphatic word, or a summary of what has been said, if the sentence is so long that it is difficult to keep the thread of meaning unbroken.

41. Clearness is increased when the beginning of the sentence prepares the way for the middle and the middle for the end, the whole forming a kind of ascent. This ascent is called "climax."

42. When the thought is expected to ascend but descends, feebleness, and sometimes confusion, is the result. The descent is called "bathos."

43. A new construction should not be introduced unexpectedly.

## Common Errors in the Use of Words

The following pages contain a short list of the more common errors in the use of words. Such a list might be extended almost indefinitely. It is only attempted to call attention to such mistakes as are, for various reasons, most liable to occur.

*A* should be repeated for every individual. "A red and black book" means one book, "a red and a black book" means two.

*Abbreviate*, and *abridge*; *abbreviation* is the shortening of a piece of writing no matter how accomplished. An *abridgement* is a condensation.

*Ability*, power to do something, should be distinguished from *capacity*, power to receive something.

*Above* should not be used as an adjective, e. g., "The statement made in *above* paragraph." Substitute *preceding*, *foregoing*, or some similar adjective.

*Accept*, not *accept of*.

*Accredit*, to give one credentials should be distinguished from *credit*, to believe what one says.

*Administer* is often misused. One *administers* a dose of medicine, the laws, an oath, or the government; one does not *administer* a blow.

*Administer to* is often incorrectly used for *minister to*, e. g., "The red cross nurse *administers to* the wounded."

*Admire* should not be used to express delight, as in the phrase "I should *admire* to do so."

*Admit* should be distinguished from *confess*.

*Advent* should be distinguished from *arrival*, *advent* meaning an epoch-making *arrival*.

*Affable* means "easy to speak to" and should not be confused with *agreeable*.

*Affect* should be distinguished from *effect*. To *affect* is to influence; to *effect* is to cause or bring about.

*Aggravate* should not be used for *annoy* or *vex* or *provoke*. It means "to make worse."

*Ain't* is a corruption of *am not*. It is inelegant though grammatical to say I *ain't* but absolutely incorrect in other persons and numbers.

*Alike* should not be accompanied by *both* as in the phrase "They are *both alike* in this respect."

*All*, *All right* should never be written *alright*. *All* and *universally* should never be used together. *All* should not be accompanied by *of*, e. g., "He received *all of* the votes." Be careful about the use of *all* in negative statements. Do not say "All present are not printers" when you mean "Not all present are printers." The first statement means there are no printers present, the second means there are some printers present.

*Allege* is a common error for *say*, *state*, and the like. It means "to declare," "to affirm," or "to assert with the idea of positiveness" and is not applicable to ordinary statements not needing emphasis.

*Allow* means *permit*, never *think* or *admit*.

*Allude to* is not the same as *mention*. A person or thing alluded to is not mentioned but indirectly implied.

*Alone* which means *unaccompanied* should be distinguished from *only* which means *no other*.

*Alternative* should never be used in speaking of more than two things.

*Altogether* is not the same as *all together*.

*Among* should not be used with *one another*, e. g., "They divided the spoil *among one another*." It should be "among themselves."

*And* should not be placed before a relative pronoun in such a position as to interfere with the construction. It should not be substituted for *to* in such cases as "Try *and* take more exercise."

*And which* should not be used for *which*.

*Another* should be followed by *than* not *from*, e. g., "Men of another temper *from* (*than*) the Greeks."

*Answer* is that which is given to a question; *reply* to an assertion.

*Anticipate* should not be used in the sense of *expect*. It means "to forestall."

*Anxious* should not be confused with *desirous*. It means "feeling anxiety."

*Any* is liable to ambiguity unless it is used with care. "Any of them" may be either singular or plural. "It is not intended for *any* machine" may mean "There is no machine for which it is intended," or "It is not intended for every machine, but only for a special type."

*Anybody else's*, idiomatic and correct.

*Anyhow*, bad, do not use it.

*Apparently* is used of what seems to be real but may not be so. It should not be confused with *evidently* which is used of what both seems to be and is real.

*Appear* is physical in its meaning and should be distinguished from *seem* which expresses a mental experience. "The forest *appears* to be impenetrable," "This does not *seem* to me to be right."

*Apt* means "skilful" and should never be used in place of *likely* or *liable*. It also means "having a natural tendency."

*As* should not be used as a causal conjunction, e. g., "Do not expect me *as* I am too uncertain of my time." The word *as* stands here as a contraction of inasmuch. Substitute a semicolon, or make two sentences.

*As to* is redundant in such expressions as "*As to* how far we can trust him I cannot say."

*At* is often incorrectly used for *in*, e. g., "He lives *at* Chicago." It is also improperly used in such expressions as "Where is he *at*?"

*As that* should not be used for *that* alone. Do not say "So *as that* such and such a thing may happen."

*Audience* is not the same as *spectators*. An *audience* listens; *spectators* merely see. A concert has an *audience*; a moving picture show has *spectators*.

*Aught* means "anything" and should not be confused with *naught* or the symbol *0* which means "nothing."

*Avenge* means to redress wrongs done to others; *revenge* wrong done to ourselves. *Avenge* usually implies just retribution. *Revenge* may be used of malicious retaliation.

*Avocation* should not be confused with *vocation*. A man's *vocation* is his principal occupation. His *avocation* is his secondary occupation.

*Aware* is not the same as *conscious*. We are *aware* of things outside of ourselves; we are *conscious* of sensations or things within ourselves.

*Awful* and *awfully* are two very much abused words. They mean "awe inspiring" and should never be used in any other sense.

*Badly* should not be used for *very much*. It should not be confused with the adjective *bad*. "He looks badly" means he makes a bad use of his eyes, say "He looks bad."

*Bank on* is slang. Say *rely on* or *trust in*.

*Beg* is often incorrectly used in the sense of *beg leave*, not "I *beg* to say" but "I *beg leave* to say."

*Beside*, meaning "by the side of" should not be confused with *besides* meaning "in addition to."

*Between* applies only to two persons or things.

*Blame on* as a verb should never be used.

*Both*, when *both—and* are used be sure they connect the right words, "He can both spell and punctuate" not "He both can spell and punctuate." Do not use such expressions as "They both resemble each other." Be careful to avoid confusion in the use of negative statements. Do not say "Both cannot go" when you mean that one can go.

*Bound* in the sense of *determined* is an Americanism and is better avoided. We say "he is *bound* to do it" meaning "he is *determined* to do it," but the phrase really means "He is under bonds, or obligation to do it."

*Bring* should be carefully distinguished from *fetch*, *carry* and *take*. *Bring* means to transfer toward the speaker. *Fetch* means to go and bring back. *Carry* and *take* mean to transfer from the speaker, e. g., "*Bring* a book home from the library." "*Fetch* me a glass of water." "*Carry* this proof to the proofreader." "*Take* this book home."

*But* is sometimes used as a preposition and when so used takes the objective case. "The boy stood on the burning deck whence all *but* him had fled." *But* should not be used in connection with *that* unless intended to express the opposite of what the meaning would be without it, e. g., "I have no doubt *but that* he will die" is incorrect because his death is expected. "I have no fear *but that* he will come" is correct, as the meaning intended is "I am sure he will come."

*But what* is often incorrectly used for *but that*. "I cannot believe *but what* he is guilty" probably means "I can but believe that he is guilty." "I *cannot but* believe" means "I must believe."

*Calculate* does not mean *think* or *suppose*.

*Calculated* does not mean *likely*. It means "intended or planned for the purpose."

*Can* which indicates ability is to be distinguished from *may* which indicates permission.

*Cannot but* should be carefully distinguished from *can but*, e. g., "I *can but* try" means "All I can do is try." "I *cannot but try*" means "I cannot help trying."

*Can't seem* should not be used for *seem unable*, e. g., "I *can't seem* to see it."

*Childlike* should be carefully distinguished from *childish*. *Childish* refers particularly to the weakness of the child.

*Come* should not be confused with *Go*. *Come* denotes motion toward the speaker; *go* motion from the speaker, "If you will come to see me, I will go to see you."

*Common* should be distinguished from *mutual*. *Common* means "shared in common." *Mutual* means "reciprocal" and can refer to but two persons or things. A *common* friend is a friend two or more friends have in common. *Mutual* friendship is the friendship of two persons for each other.

*Compare to*, *liken to*, *compare with*, means "measure by" or "point out similarities and differences."

*Condign* means "suitable" or "deserved," not necessarily *severe*.

*Condone* means "to forgive" or "nullify by word or act," not "make amends for."

*Consider* in the sense of *regard as* should not usually be followed by *as*, e. g., "I consider him a wise man," not "*as* a wise man."

*Contemptible* is used of an object of contempt and it should be distinguished from *contemptuous* which is used of what is directed at such an object, e. g., "He is a *contemptible* fellow." "I gave him a *contemptuous* look."

*Continual* should not be confused with *continuous*. *Continual* means "frequently repeated." *Continuous* means "uninterrupted."

*Convene*, which means "to come together," should not be confused with *convoke* which means "to bring or call together." A legislature *convenes*. It cannot be *convened* by another, but it can be *convoked*.

*Crime* is often used for offenses against the speaker's sense of right. Properly *crime* is a technical word meaning "offenses against law." A most innocent action may be a *crime* if it is contrary to a statute. The most sinful, cruel, or dishonest action is no *crime* unless prohibited by a statute.

*Dangerous* should not be used for *dangerously ill*.

*Data* is plural.

*Deadly*, "that which inflicts death" should not be confused with *deathly*, "that which resembles death."

*Decided* must not be confused with *decisive*. A *decided* victory is a clear and unmistakable victory. A *decisive* victory is one which decides the outcome of a war or of a campaign.

*Decimate* means to take away one-tenth. It is not properly used in a general way of the infliction of severe losses.

*Definite* which means "well defined" should not be confused with *definitive* which means "final."

*Demean* is related to *demeanor* and means "behave." It should be carefully distinguished from *degrade* or *lower*.

*Die.* We die *of* a certain disease, not *with* or *from* it.

*Differ* in the sense of disagree is followed by *with*. "I *differ with* you." *Differ* as indicating unlikeness is followed by *from*.

*Different* should be followed by *from* never by *with*, *than*, or *to*.

*Directly* should not be used for *as soon as*.

*Discover*, "to find something which previously existed" should be distinguished from *invent* something for the first time.

*Disinterested* means "having no financial or material interest in a thing." It should be carefully distinguished from *uninterested* which means "taking no interest in" a thing.

*Dispense*, "to distribute" should not be confused with *dispense with*, "to do without."

*Disposition* is not the same as *disposal*.

*Distinguish* which means "to perceive differences" should not be confused with *differentiate* which means "to make or constitute a difference."

*Divide* should be carefully distinguished from *distribute*.

*Don't* is a contraction of do not. *Doesn't* is the contraction for does not. *I don't*, *they don't*, *he doesn't*.

*Due* should not be used for *owing to* or *because of*.

*Each* is distributive and is always singular. *Each other* which is applicable to two only should not be confused with *one another* which is applicable to more than two.

*Egotist*, a man with a high or conceited opinion of himself, should not be confused with *egoist* which is the name for a believer in a certain philosophical doctrine.

*Either* is distributive and therefore singular and should never be used of more than two.

*Elegant* denotes delicacy and refinement and should not be used as a term of general approval.

*Else* should be followed by *than*, not by *but*. "No one else *than* (not *but*) he could have done so much."

*Emigrant*, one who goes out of a country should not be confused with *immigrant*, one who comes into a country.

*Enormity* is used of wickedness, cruelty, or horror, not of great size, for which *enormousness* should be used. We speak of the *enormity* of an offence but of the *enormousness* of a crowd.

*Enthuse* should not be used as a verb.

*Equally as* well; say *equally well*, or *as well*.

*Every place* used adverbially should be *everywhere*.

*Except* should never be used in the sense of *unless* or *but*.

*Exceptional* which means "unusual," "forming an exception" should not be confused with *exceptionable* which means "open to objection."

*Expect* which involves a sense of the future should not be confused with *suppose* and similar words, as in the phrase "I *expect* you know all about it."

*Factor* is not to be confounded with *cause*.

*Falsity* applies to things, *falseness* to persons.

*At fault* means "at a loss of what to do next." *In fault* means "in the wrong."

*Favor* should not be used in the sense of *resemble*.

*Female* should not be used for *woman*. The words *female*, *woman*, and *lady* should be used with careful attention to their respective shades of meaning.

*Few*, which emphasizes the fact that the number is small should be distinguished from *a few* which emphasizes the fact that there is a number though it be small. "*Few* shall part where many meet." "*A few* persons were saved in the ark."

*Fewer* applies to number; *less* to quantity.

*Firstly* should not be used for *first* although secondly and thirdly may be used to complete the series.

*Fix* should not be used in the sense of *repair*, *arrange*, or *settle*.

*Former* and *latter* should never be used where more than two things are involved.

*Frequently* should be distinguished from commonly, *generally*, *perpetually*, *usually*. *Commonly* is the antithesis of *rarely*, *frequently* of *seldom*, *generally* of *occasionally*, *usually* of *casually*.

*Funny* should not be used to mean *strange* or *remarkable*.

*Gentleman Friend* and *Lady Friend* are expressions which should be avoided, say "man or woman friend" or "man or woman of my acquaintance" or even "gentleman or lady of my acquaintance."

*Good* should not be used in the sense of *well*. "I feel *good*."

*Got* is said to be the most misused word in the language. The verb means to secure by effort and should be used only with this meaning, e. g., "I have *got* the contract." *Have got* to indicate mere possession is objectionable. Mere possession is indicated by *have* alone. Another common mistake is the use of *got* to express obligation or constraint. "I have *got* to do it."

*Guess* should not be used in the sense of *think* or *imagine*.

*Handy* should never be used to express nearness.

*Hanged* should be used to express the execution of a human being. *Hung* is the past participle in all other uses.

*Hardly.* "I *can hardly* see it," not "I *can't hardly* see it."

*Healthy* which means "possessed of health" should be distinguished from *healthful* and *wholesome* which mean "health giving."

*High* should not be confused with *tall*.

*Home* is not a synonym for *house*. A beautiful *house* is a very different thing from a beautiful *home*.

*Honorable* as a title should always be preceded by *the*.

*How* should not be used for *what*, or for *that*. It means "in what manner."

*How that* should not be used when either one will do alone. Such a sentence as "We have already noted how that Tillotson defied rubrical order...." is very bad.

*If* should not be used in the sense of *where* or *that*.

*Ilk* means "the same" not *kind* or *sort*.

*Ill* is an adverb as well as an adjective. Do not say illy.

*In* should not be used for *into* when motion is implied. You ride *in* a car but you get *into* it.

*Inaugurate* should not be used for *begin*.

*Individual* should not be used for *person*.

*Inside of* should not be used as an expression of time.

*Invaluable*, meaning "of very great value" should not be confused with *valueless*, meaning "of no value."

*Invite* should not be used for *invitation*.

*Kind* is not plural. Do not say "These" or "those" *kind* of things. *Kind of* should never be followed by the indefinite article. "What *kind of* man is he?" not "What *kind of a* man is he?" *Kind of* or *sort of* should not be used in the sense of *rather* or *somewhat*.

*Kindly* is often misused in such expressions as "You are *kindly* requested to recommend a compositor." Undoubtedly the idea of kindness is attached to the recommendation not to the request and the sentence should be so framed as to express it.

*Last* is often misused for *latest*. "The *last* number of the paper" is not the one that appeared this morning but the one that finally closes publication.

*Latter* applies only to the last of two. If a longer series than two is referred to, say *the last*.

*Lay*, which is a transitive verb, should not be confused with *lie*. *Lay* is a verb which expresses causitive action; *lie* expresses passivity. "He *lays* plans." "He *lies* down." The past tense of *lay* is *laid*, that of *lie* is *lay*.

*Learn* should not be used in place of *teach*.

*Lengthy* is a very poor substitute for *long*, which needs no substitute.

*Liable* should not be used for *likely*. *Liable* means an unpleasant probability. *Likely* means any probability. *Liable* is also used to express obligation. He is *liable* for this debt.

*Like* must never be used in the sense of *as*. "Do *like* I do" should be "Do *as* I do."

*Literally* implies that a statement to which it is attached is accurately and precisely true. It is frequently misused.

*Loan* is a noun, not a verb.

*Locate* should not be used in the sense of *settle*.

*Lot* or *lots* should not be used to indicate a *great deal*.

*Love* expresses affection or, in its biblical sense, earnest benevolence. *Like* expresses taste. Do not say "I should *love* to go."

*Lovely* means "worthy of affection" and, like *elegant*, should never be used as a term of general approbation.

*Luxuriant* which means "superabundant in growth or production" should not be confounded with *luxurious* which means "given over to luxury." Vegetation is *luxuriant*, men are *luxurious*.

*Mad* means *insane* and is not a synonym for *angry*.

*Means* may be either singular or plural.

*Meet* should not be used in the sense of *meeting* except in the case of a few special expressions such as "a race meet."

*Mighty* should not be used in the sense of *very*.

*Mind* should not be used in the sense of *obey*.

*Minus* should not be used in the sense of *without* or *lacking*.

*Most* should not be used instead of *almost*, as in such expressions as "It rained *most* every day."

*Must* should not be used for *had to* or *was obliged*. In its proper use it refers to the present or future only.

*Necessities* should be carefully distinguished from *necessaries*.

*Negligence*, which denotes a quality of character should be distinguished from *neglect* which means "a failure to act."

*Neither* denotes one of two and should not be used for *none* or *no one*. As a correlative conjunction it should be followed by *nor* never by *or*.

*New beginner*. *Beginner* is enough; all beginners are new.

*News* is singular in construction.

*Never* is sometimes used as an emphatic negative but such usage is not good.

*Nice* should not be used in the sense of *pleasant* or *agreeable*.

*No how* should not be used for *anyway*.

*No place* should be written as *nowhere*.

*None* should be treated as a singular.

*Not*, like *neither*, must be followed by the correlative *nor*, e. g., "Not for wealth nor for fame did he strive."

*Not* ... *but* to express a negative is a double negative and therefore should not be used, e. g., "I have *not* had *but* one meal to-day."

*Nothing like* and *nowhere near* should not be used for *not nearly*.

*O* should be used for the vocative and without punctuation.

*Oh* should be used for the ejaculation and should be followed by a comma or an exclamation point.

*Obligate* should not be used for *oblige*.

*Observe* should not be used for *say*.

*Observation* should not be used for *observance*.

*Of* is superfluous in such phrases as *smell of*, *taste of*, *feel of*.

*Off* should never be used with *of*; one or the other is superfluous.

*Other*. After *no other* use *than*, not *but*.

*Ought* must never be used in connection with *had* or *did*. "You *hadn't ought* or *didn't ought* to do it" should be "You ought not to have done it."

*Out loud* should never be used for *aloud*.

*Panacea* is something that cures all diseases, not an effective remedy for one disease.

*Partake of* should not be used in the sense of *eat*. It means "to share with others."

*Party* should never be used for *person* except in legal documents.

*Per* should be used in connection with other words of Latin form but not with English words. *Per diem*, *per annum*, and the like are correct. *Per day* or *per year* are incorrect. It should be *a day*, or *a year*.

*Perpendicular*, which merely means at right angles to something else mentioned, should not be used for *vertical*.

*Plenty*, a noun should not be confused with the adjective *plentiful*.

*Politics* is singular.

*Post* does not mean *inform*.

*Predicate* should not be used in the sense of *predict* or in the sense of *base* or *found*.

*Premature* means "before the proper time." It should not be used in a general way as equivalent to *false*.

*Pretty* should not be used in the modifying sense, nor as a synonym for *very* in such phrases as "pretty good," "pretty near," and the like.

*Preventative*, no such word, say *preventive*.

*Promise* should not be used in the sense of *assure*.

*Propose*, meaning "to offer" should not be confused with *purpose* meaning "to intend."

*Proposition* should not be confounded with *proposal*. A *proposition* is a statement of a statement or a plan. A *proposal* is the presentation or statement of an offer.

*Providing* should not be used for *provided*.

*Quality* should never be used as an adjective or with an adjective sense. "Quality clothes" is meaningless: "Clothes of quality" equally so. All clothes have quality and the expression has meaning only when the quality is defined as good, bad, high, low, and so forth.

*Quit*, "to go away from" is not the same as *stop*.

*Quite* means "entirely," "wholly," and should never be used in the modifying sense as if meaning *rather* or *somewhat*. "Quite a few" is nonsense.

*Raise* is a much abused word. It is never a noun. As a verb it should be distinguished from *rear* and *increase*, as in such phrases as "He was *raised* in Texas." "The landlord *raised* my rent."

*Rarely ever* should not be used for *rarely* or *hardly ever*.

*Real* should not be used in the sense of *very*.

*Reference* should be used with *with* rather than *in*. Say *with* reference to, not *in* reference to. The same rule applies to the words *regard* and *respect*. Do not say "*in regards to*," say "*with regard to*."

*Remember* is not the same as *recollect*, which means "to remember by an effort."

*Rendition* should not be used for *rendering*.

*Researcher* has no standing as a word.

*Reside* in the sense of live, and residence in the sense of house or dwelling are affectations and should never be used.

*Retire* should not be used in the sense of "go to bed."

*Right* should not be used in the sense of *duty*. "You *had a right* to warn me," should be "It was your duty to warn me, or you ought to have warned me." *Right* should not be used in the sense of *very*. Such expressions as *right now*, *right off*, *right away*, *right here* are not now in good use.

*Same* should not be used as a pronoun. This is a common usage in business correspondence but it is not good English and can be easily avoided without sacrificing either brevity or sense. *Same as* in the sense of *just as*, *in the same manner* should be avoided.

*Score* should not be used for *achieve* or *accomplish*.

*Set* should not be confused with *sit*. To set means "to cause to sit."

*Sewage*, meaning the contents of a sewer, should not be confused with *sewerage* which means the system.

*Show* should not be used in the sense of *play* or *performance*. *Show up* should not be used for *expose*.

*Since* should not be used for *ago*.

*Size up* should not be used for *estimate* or *weigh*.

*Some* should not be used for *somewhat* as "I feel *some* better."

*Sort of* should not be used for *rather*.

*Splendid* means *shining* or *brilliant* and should not be used as a term of general commendation.

*Stand for* means "be responsible for." Its recent use as meaning *stand*, *endure*, or *permit*, should be avoided.

*Start* should not be used for *begin*, e. g., "He *started* (began) to speak."

*State* should not be used for *say*.

*Stop* should not be used for *stay*.

*Such* should not be used for *so*. Say "I have never seen *so* beautiful a book before" not "I have never seen such a beautiful book before."

*Sure* should not be used as an adverb. Say *surely*.

*Take* is superfluous in connection with other verbs, e. g., "Suppose we *take* and *use* that type." *Take* should not be confused with *bring*. *Take stock in* should not be used for *rely* or *trust in*.

*That* should not be used in the sense of *so*. "I did not know it was *that* big."

*Think* should not have the word *for* added, e. g., "It is more important than you *think for*."

*This* should not be used as an adverb. "This much is clear" should be "Thus much is clear."

*Through* should not be used for *finished*.

*To* is superfluous and wrong in such expressions as "Where did you go *to*?"

*Too* alone should not modify a past participle. "He was *too* (much) excited to reply."

*Transpire* does not mean *happen*. It means to come to light or become known.

*Treat* should be followed by *of* rather than *on*. This volume treats *of* grammar, not *on* grammar.

*Try* should be followed by *to* rather than *and*. "I will try *to* go," not "I will try *and* go."

*Ugly* should never be used in the sense of *bad tempered* or *vicious*. It means "repulsive to the eye."

*Unique* does not mean *rare*, *odd*, or *unusual*. It means alone of its kind.

*Upward of* should not be used in the sense of *more than*.

*Venal* should not be confused with *venial*.

*Verbal* should not be confused with *oral*. A *verbal* message means only a message in words; an *oral* message is a message by word of mouth.

*Very* should be used sparingly. It is a word of great emphasis and like all such words defeats its purpose when used too frequently.

*Visitor* is a human caller. *Visitant* a supernatural caller.

*Want* should not be used in the sense of *wish*, e. g., "I *want* it" really means "I feel the want of it" or "I lack it." *Want*, *wish*, and *need* should be carefully distinguished.

*Way* should not be used in the sense of *away* in such expressions as "*Way* down East."

*Ways* should not be used for *way*, e. g., "It is quite a *ways* (way) off."

*What* is often misused for *that*, e. g., "He has no doubt but *what* (that) he will succeed."

*Whence* means "from what place or cause" and should not be preceded by *from*. This applies equally to hence which means "from this place."

*Which* should not be used with a clause as its antecedent, e. g., "He replied hotly, *which* was a mistake" should be "He replied hotly; this was a mistake." *Which* being a neuter pronoun should not be used to represent a masculine or feminine noun. Use who. Between the two neuter pronouns *which* and *that* let euphony decide.

*Who* should not be misused for *whom* or *whose*, e. g., "*Who* (whom) did you wish to see?" "Washington, than *who* (whose) no greater name is recorded." Impersonal objects should be referred to by *which* rather than *who*.

*Without* should not be used for *unless*, e. g., "I will not go *without* (unless) you go with me."

*Witness* should not be used for *see*.

*Worst kind* or *worst kind of way* should not be used for *very much*.

*Womanly* means "belonging to woman as woman."

*Womanish* means *effeminate*.

## Tables of Irregular Verbs

Table 1 contains the principal parts of all irregular verbs whose past tense and perfect participle are unlike.

Most errors in the use of irregular verbs occur with those in Table 1. The past tense must not be used with *have* (*has*, *had*). Do not use such expressions as *have drove* and *has went*. Equally disagreeable is the use of the perfect participle for the past tense; as, *she seen*, *they done*.

### Table I 

+---------------------+-------------------+-------------------+-------------------+-------------------+
| Present Tense       |                   | Past Tense        |                   | Perf. Part.       |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| arise               |                   | arose             |                   | arisen            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| be or am            |                   | was               |                   | been              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| bear, *bring forth* |                   | bore              |                   | born, borne       |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| bear, *carry*       |                   | bore              |                   | borne             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| beat                |                   | beat              |                   | beaten, beat      |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| begin               |                   | began             |                   | begun             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| bid                 |                   | bade, bid         |                   | bidden, bid       |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| bite                |                   | bit               |                   | bitten, bit       |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| blow                |                   | blew              |                   | blown             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| break               |                   | broke             |                   | broken            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| chide               |                   | chid              |                   | chidden, chid     |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| choose              |                   | chose             |                   | chosen            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| cleave, *split*     | {                 | cleft, clove      | {                 | cleft, cleaved,   |
|                     |                   | (clave)           |                   | cloven            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| come                |                   | came              |                   | come              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| do                  |                   | did               |                   | done              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| draw                |                   | drew              |                   | drawn             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| drink               |                   | drank             |                   | drunk, drunken    |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| drive               |                   | drove             |                   | driven            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| eat                 |                   | ate (eat)         |                   | eaten (eat)       |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| fall                |                   | fell              |                   | fallen            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| fly                 |                   | flew              |                   | flown             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| forbear             |                   | forbore           |                   | forborne          |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| forget              |                   | forgot            |                   | forgotten, forgot |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| forsake             |                   | forsook           |                   | forsaken          |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| freeze              |                   | froze             |                   | frozen            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| give                |                   | gave              |                   | given             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| go                  |                   | went              |                   | gone              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| grow                |                   | grew              |                   | grown             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| hide                |                   | hid               |                   | hidden, hid       |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| know                |                   | knew              |                   | known             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| lie, *recline*      |                   | lay               |                   | lain              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| ride                |                   | rode              |                   | ridden            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| ring                |                   | rang, rung        |                   | rung              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| rise                |                   | rose              |                   | risen             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| run                 |                   | ran               |                   | run               |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| see                 |                   | saw               |                   | seen              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| shake               |                   | shook             |                   | shaken            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| shrink              |                   | shrank, shrunk    |                   | shrunk, shrunken  |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| sing                |                   | sung, sang        |                   | sung              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| sink                |                   | sank, sunk        |                   | sunk              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| slay                |                   | slew              |                   | slain             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| slide               |                   | slid              |                   | slidden, slid     |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| smite               |                   | smote             |                   | smitten           |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| speak               |                   | spoke (spake)     |                   | spoken            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| spring              |                   | sprang, spring    |                   | sprung            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| steal               |                   | stole             |                   | stolen            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| stride              |                   | strode            |                   | stridden          |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| strike              |                   | struck            |                   | struck, stricken  |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| strive              |                   | strove            |                   | striven           |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| swear               |                   | swore (sware)     |                   | sworn             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| swim                |                   | swam, swum        |                   | swum              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| take                |                   | took              |                   | taken             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| tear                |                   | tore              |                   | torn              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| throw               |                   | threw             |                   | thrown            |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| tread               |                   | trod              |                   | trodden, trod     |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| wear                |                   | wore              |                   | worn              |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| weave               |                   | wove              |                   | woven             |
+---------------------+-------------------+-------------------+-------------------+-------------------+
| write               |                   | wrote             |                   | written           |
+---------------------+-------------------+-------------------+-------------------+-------------------+

### Table II

This table contains the principal parts of all irregular verbs whose past tense and perfect participles are alike.

  --------------- ---------- ----------------- ---------- --------------- ---------- -----------------
  Present Tense              Past Tense and                Present Tense              Past Tense and  
                             Perf. Part.                                             Perf. Part.

  abide                      abode                        mean                       meant

  behold                     beheld                       meet                       met

  beseech                    besought                     pay                        paid

  bind                       bound                        put                        put

  bleed                      bled                         read                       read

  breed                      bred                         rend                       rent

  bring                      brought                      say                        said

  build                      built                        seek                       sought

  burst                      burst                        sell                       sold

  buy                        bought                       send                       sent

  cast                       cast                         set                        set

  catch                      caught                       shed                       shed

  cling                      clung                        shoe                       shod

  cost                       cost                         shoot                      shot

  creep                      crept                        shut                       shut

  cut                        cut                          sit                        sat

  deal                       dealt                        sleep                      slept

  feed                       fed                          sling                      slung

  feel                       felt                         slink                      slunk

  fight                      fought                       spend                      spent

  find                       found                        spin                       spun (span)

  flee                       fled                         spit                       spit (spat)

  fling                      flung                        split                      split

  get                        got (gotten)                 spread                     spread

  grind                      ground                       stand                      stood

  have                       had                          stick                      stuck

  hear                       heard                        sting                      stung

  hit                        hit                          string                     strung

  hold                       held                         sweep                      swept

  hurt                       hurt                         swing                      swung

  keep                       kept                         teach                      taught

  lay                        laid                         tell                       told

  lead                       led                          think                      thought

  leave                      left                         thrust                     thrust

  lend                       lent                         weep                       wept

  let                        let                          win                        won

  lose                       lost                         wring                      wrung

  make                       made                                                    
  --------------- ---------- ----------------- ---------- --------------- ---------- -----------------

### Table III

This table includes verbs that are both regular and irregular.

#### A

Verbs in which the regular form is preferred.

  ------------------------- -------------- ---------------------- -------------- ----------------------
  Present Tense                            Past Tense                            Perf. Part.

  bend                                     bended, bent                          bended, bent

  bereave                                  bereaved, bereft                      bereaved, bereft

  blend                                    blended, blent                        blended, blent

  bless                                    blessed, blest                        blessed, blest

  burn                                     burned, burnt                         burned, burnt

  cleave, *stick*                          cleaved (clave)                       cleaved

  clothe                                   clothed, clad                         clothed, clad

  curse                                    cursed, curst                         cursed, curst

  dive                                     dived (dove)                          dived (dove)

  dream                                    dreamed, dreamt                       dreamed, dreamt

  dress                                    dressed, drest                        dressed, drest

  gild                                     gilded, gilt                          gilded, gilt

  heave                                    heaved, hove                          heaved, hove

  hew                                      hewed                                 hewed, hewn

  lade                                     laded                                 laded, laden

  lean                                     leaned, leant                         leaned, leant

  leap                                     leaped, leapt                         leaped, leapt

  learn                                    learned, learnt                       learned, learnt

  light                                    lighted, lit                          lighted, lit

  mow                                      mowed                                 mowed, mown

  pen, *shut up*                           penned, pent                          penned, pent

  plead                                    pleaded (plead *or*                   pleaded (plead *or* 
                                           pled)                                 pled)

  prove                                    proved                                proved, proven

  reave                                    reaved, reft                          reaved, reft

  rive                                     rived                                 rived, riven

  saw                                      sawed                                 sawed, sawn

  seethe                                   seethed (sod)                         seethed, sodden

  shape                                    shaped                                shaped, shapen

  shave                                    shaved                                shaved, shaven

  shear                                    sheared                               sheared, shorn

  smell                                    smelled, smelt                        smelled, smelt

  sow                                      sowed                                 sowed, sown

  spell                                    spelled, spelt                        spelled, spelt

  spill                                    spilled, spilt                        spilled, spilt

  spoil                                    spoiled, spoilt                       spoiled, spoilt

  stave                                    staved, stove                         staved, stove

  stay                                     stayed, staid                         stayed, staid

  swell                                    swelled                               swelled, swollen

  wake                                     waked, woke                           waked, woke

  wax, *grow*                              waxed                                 waxed (waxen)

  wed                                      wedded                                wedded, wed

  whet                                     whetted, whet                         whetted, whet

  work                                     worked, wrought                       worked, wrought
  ------------------------- -------------- ---------------------- -------------- ----------------------

#### B

Verbs in which the irregular form is preferred.

|               |     |                          |     |                  |
|---------------|-----|--------------------------|-----|------------------|
| Present Tense |     | Past Tense               |     | Perf. Part.      |
| awake         |     | awoke, awaked            |     | awaked, awoke    |
| belay         |     | belaid, belayed          |     | belaid, belayed  |
| bet           |     | bet, betted              |     | bet, betted      |
| crow          |     | crew, crowed             |     | crowed           |
| dare          |     | durst, dared             |     | dared            |
| dig           |     | dug, digged              |     | dug, digged      |
| dwell         |     | dwelt, dwelled           |     | dwelt, dwelled   |
| gird          |     | girt, girded             |     | girt, girded     |
| grave         |     | graved                   |     | graven, graved   |
| hang          |     | hung, hanged             |     | hung, hanged     |
| kneel         |     | knelt, kneeled           |     | knelt, kneeled   |
| knit          |     | knit, knitted            |     | knit, knitted    |
| quit          |     | quit, quitted            |     | quit, quitted    |
| rap           |     | rapt, rapped             |     | rapt, rapped     |
| rid           |     | rid, ridded              |     | rid, ridded      |
| shine         |     | shone (shined)           |     | shone (shined)   |
| show          |     | showed                   |     | shown, showed    |
| shred         |     | shred, shredded          |     | shred, shredded  |
| shrive        |     | shrived, shrove          |     | shriven, shrived |
| slit          |     | slit, slitted            |     | slit, slitted    |
| speed         |     | sped, speeded            |     | sped, speeded    |
| strew         |     | strewed                  |     | strewn, strewed  |
| strow         |     | strowed                  |     | strown, strowed  |
| sweat         |     | sweat, sweated           |     | sweat, sweated   |
| thrive        |     | throve, thrived          |     | thrived, thriven |
| wet           |     | wet (wetted)             |     | wet (wetted)     |
| wind          |     | wound (winded)           |     | wound (winded)   |

The verbs of the following list also are irregular; but as they lack one or more of the principal parts, they are called defective verbs.

## Defective Verbs

|         |     |        |     |          |     |           |
|---------|-----|--------|-----|----------|-----|-----------|
| Present |     | Past   |     | Present  |     | Past      |
| can     |     | could  |     | ought    |     | .....     |
| may     |     | might  |     | .....    |     | quoth     |
| must    |     | .....  |     | beware   |     | .....     |
| shall   |     | should |     | methinks |     | methought |
| will    |     | would  |     |          |     |           |

All the participles are wanting in defective verbs.

The verb *ought*, when used to express past duty or obligation, is followed by what is called the perfect infinitive—a use peculiar to itself because *ought* has no past form.

*Example:* I ought *to have gone* yesterday.

Other verbs expressing past time are used in the past tense followed by the root infinitive.

*Example:* I intended *to go* yesterday.
"""

def edit(args):
    old = args.get("old")
    new = args.get("new")
    reason = args.get("reason")
    if (isinstance(old, str) and 
        isinstance(new, str) and
        isinstance(reason, str)):
        if old.strip() and new.strip():
            return { 'old': old, 'new': new, 'reason': reason }
    return 'error'


TOOLS = {
    "edit": (
        "Replace old with new. Give a short 60 character reason for the edit",
        {
            "type": "object",
            "properties": {
                "old": {"type": "string"},
                "new": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["old", "new", "reason" ],
        },
        edit,
    ),
}


def run_tool(name, args):
    try:
        return TOOLS[name][2](json.loads(args))
    except Exception as err:
        return f"error: {err}"


def make_schema():
    result = []
    for name, (description, params, _fn) in TOOLS.items():
        result.append({
            "type": "function",
            "function": {
                "name": name, 
                "description": description, 
                "parameters": params
            },
        })
    return result


def call_api(model, messages):
    try:
        tools = make_schema()
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=MAX_TOKENS)
        return response
    except Exception as err:
        return {"error": f"litellm call failed: {err}"}


def proofread(target, model, interactive=False):
        messages = []
        suggestions = []

        # system prompt
        if not interactive:
             content = SYSTEM_PROMPT.strip()
             messages.append({ "role": "system", "content": content })

        # instructions
        content = GRAMMAR_RULES.strip()
        messages.append({ "role": "system", "content": content })

        # user prompt
        content = target.strip()
        messages.append({ "role": "user", "content": content })

        for iterations in range(MAX_TRIES):
            response = call_api(model, messages)
            if "error" in response:
                print(f"agent error: {response['error']}")
                break

            d = response.choices[0]
            message = d.message
            finish_reason = d.finish_reason
            tool_calls = message.tool_calls or []
            reasoning_content = getattr(message, "reasoning_content", '')
            message_content = message.content or ''

            # debug
            if reasoning_content:
                print(f'--- agent thinking\n{reasoning_content}')
            if message_content:
                print(f'--- agent message\n{message_content}')
            for tc in tool_calls:
                print(f'--- agent tool\n{tc.function.name}({tc.function.arguments})')

            # append reply
            content = message_content or reasoning_content
            m = {"role": "assistant", "content": content }
            if tool_calls:
                m["tool_calls"] = [{
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name, 
                        "arguments": tc.function.arguments
                    },
                } for tc in tool_calls ]
            messages.append(m)

            if not tool_calls:
                break

            # call tools
            for tc in tool_calls:
                result = run_tool(tc.function.name, tc.function.arguments)
                if isinstance(result, dict):
                    suggestions.append(result)
                    result = "ok"
                messages.append({
                    "role": "tool", 
                    "tool_call_id": tc.id, 
                    "content": result
                })
        print('--- agent done')
        return suggestions


def create_server(args):
    model = args.model
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    corrections = []
    future = None
    past_text = ''

    class ProofreadHandler(BaseHTTPRequestHandler):

        def do_GET(self):
            response = json.dumps([{
                "name": "English (US)", 
                "code": "en", 
                "longCode": "en-US"
            }]).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response)

        def do_POST(self):
            nonlocal future
            nonlocal past_text
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length).decode()
            params = urllib.parse.parse_qs(body)
            text = params.get('text', [''])[0]
            text = text.strip()
            print('> received text:', repr(text))

            if future and future.done():
                print('> corrections pulled.')
                corrections.extend(future.result())
                future = None

            if future is None and text and text != past_text:
                print('> submitting text.')
                future = executor.submit(proofread, text, model=model)
                past_text = text

            matches = []
            for i, d in enumerate(corrections):
                old = d['old']
                new = d['new']
                reason = d['reason']
                offset = text.find(old)
                length = len(old)
                if offset != -1:
                    matches.append({
                        "message": reason,
                        "offset": offset,
                        "length": length,
                        "replacements": [{"value": new}],
                        "rule": { "issueType": "grammar" }
                    })

            response = json.dumps({
                "software": {"name": "CustomPython"}, 
                "matches": matches
            })
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response.encode())

    print(f'Server running on http://{args.host}:{args.port}...')
    return HTTPServer((args.host, args.port), ProofreadHandler)


def main():
    parser = argparse.ArgumentParser(
        description="LanguageTool Server using LLM to check grammar",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default='127.0.0.1', help="host to serve.")
    parser.add_argument("--port", type=int, default=8081, help="port to serve.")
    parser.add_argument("--model", default=MODEL, help="prefix with provider.")
    parser.add_argument('--interactive', action='store_true', help='ask grammar questions')
    args = parser.parse_args()
    if args.interactive:
        import readline
        while True:
            prompt = input('prompt> ')
            proofread(prompt, args.model, interactive=True)
    server = create_server(args)
    server.serve_forever()


if __name__ == "__main__":
    main()


