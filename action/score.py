# sentence pair sts annotate template 
from langchain.chains import  ConversationChain
from langchain.prompts import ChatPromptTemplate   
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.schema.output_parser import OutputParserException
from retrying import retry

score_schema1 = ResponseSchema(name="1. Scoring", description="sts score for sentence pair1")
score_schema2 = ResponseSchema(name="2. Scoring", description="sts score for sentence pair2")
score_schema3 = ResponseSchema(name="3. Scoring", description="sts score for sentence pair3")
score_schema4 = ResponseSchema(name="4. Scoring", description="sts score for sentence pair4")
score_schema5 = ResponseSchema(name="5. Scoring", description="sts score for sentence pair5")
score_schema6 = ResponseSchema(name="6. Scoring", description="sts score for sentence pair6")
score_schema7 = ResponseSchema(name="7. Scoring", description="sts score for sentence pair7")
score_schema8 = ResponseSchema(name="8. Scoring", description="sts score for sentence pair8")
score_schema9 = ResponseSchema(name="9. Scoring", description="sts score for sentence pair9")
score_schema10 = ResponseSchema(name="10. Scoring", description="sts score for sentence pair10")

score_schemas = [score_schema1,score_schema2,score_schema3, score_schema4, score_schema5, \
    score_schema6, score_schema7, score_schema8, score_schema9, score_schema10]
score_parser = StructuredOutputParser.from_response_schemas(score_schemas)
score_format_instructions = score_parser.get_format_instructions()



def retry_if_output_parser_exception(exception):

    return isinstance(exception, OutputParserException)

@retry(wait_random_min=256, wait_random_max=1000*10, stop_max_attempt_number=3,retry_on_exception=retry_if_output_parser_exception)
def sts_score(llm, ans1_dict):
    #write 
    conversation_score = ConversationChain(llm=llm, verbose=True)
    score_template = '''
        Role:
        <<<You are a sentence semantic similarity score annotator. Follow the standard and score the semantic similarity score between two sentences. >>>
        
        Standard of scoring:
        <<<Here is the standard of scoring:
        The two sentences are completely equivalent, as they mean the same thing. Scoring 5
        e.g. Sentence 1: The bird is bathing in the sink. Sentence 2: Birdie is washing itself in the water basin.

        The two sentences are mostly equivalent, but some unimportant details differ. Scoring 4
        e.g. Sentence 1: Two boys on a couch are playing video games. Sentence 2: Two boys are playing a video game.

        The two sentences are roughly equivalent, but some important information differs/missing. Scoring 3
        e.g. Sentence 1: John said he is considered a witness but not a suspect. Sentence 2: "He is not a suspect anymore." John said.

        The two sentences are not equivalent, but share some details. Scoring 2
        e.g. Sentence 1: They flew out of the nest in groups. Sentence 2: They flew into the nest together.

        The two sentences are not equivalent, but are on the same topic. Scoring 1
        e.g. Sentence 1: The woman is playing the violin. Sentence 2: The young lady enjoys listening to the guitar.

        The two sentences are completely dissimilar. Scoring 0
        e.g. Sentence 1: The black dog is running through the snow. Sentence 2: A race car driver is driving his car through the mud.>>>

        Demand: 
        <<<1. Learn from these examples.
        2. Pay attention to the changes in relationships that cause shifts in meaning and the changes in voice that keep the meaning unchanged.
        3. Carefully analyze case by case. Don't be influenced by each other. >>> Please adhere to these rules.

        Your case:
        <<<
        1. Sentence 1: {sent1}    Sentence 2: {pair1}
        2. Sentence 1: {sent2}    Sentence 2: {pair2}
        3. Sentence 1: {sent3}    Sentence 2: {pair3}
        4. Sentence 1: {sent4}    Sentence 2: {pair4}
        5. Sentence 1: {sent5}    Sentence 2: {pair5}
        6. Sentence 1: {sent6}    Sentence 2: {pair6}
        7. Sentence 1: {sent7}    Sentence 2: {pair7}
        8. Sentence 1: {sent8}    Sentence 2: {pair8}
        9. Sentence 1: {sent9}    Sentence 2: {pair9}
        10. Sentence 1: {sent10}    Sentence 2: {pair10}

        Your answer:
        {score_format_instructions}
        >>>

    '''
    score_prompt = ChatPromptTemplate.from_template(template=score_template)
    prompt_s1 = score_prompt.format_messages(sent1 = ans1_dict['1.'],sent2 = ans1_dict['2.'],sent3 = ans1_dict['3.'],sent4 = ans1_dict['4.'],sent5 = ans1_dict['5.'],\
        sent6 = ans1_dict['6.'],sent7 = ans1_dict['7.'],sent8 = ans1_dict['8.'],sent9 = ans1_dict['9.'],sent10 = ans1_dict['10.'],\
        pair1 = ans1_dict['Change1:'], pair2 = ans1_dict['Change2:'], pair3 = ans1_dict['Change3:'], pair4 = ans1_dict['Change4:'], pair5 = ans1_dict['Change5:'],\
        pair6 = ans1_dict['Change6:'], pair7 = ans1_dict['Change7:'], pair8 = ans1_dict['Change8:'], pair9 = ans1_dict['Change9:'], pair10 = ans1_dict['Change10:'],\
        score_format_instructions=score_format_instructions)
    ans3 = conversation_score.predict(input = prompt_s1[0].content)
    ans3_dict = score_parser.parse(ans3)
    # print(ans3)
    return ans3_dict
