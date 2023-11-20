# sentence write check template 
from langchain.chains import  ConversationChain
from langchain.prompts import ChatPromptTemplate   
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.schema.output_parser import OutputParserException
from retrying import retry

origin1 = ResponseSchema(name="1.",description="load the origin1")
origin2 = ResponseSchema(name="2.",description="load the origin2")
origin3 = ResponseSchema(name="3.",description="load the origin3")
origin4 = ResponseSchema(name="4.",description="load the origin4")
origin5 = ResponseSchema(name="5.",description="load the origin5")
origin6 = ResponseSchema(name="6.",description="load the origin6")
origin7 = ResponseSchema(name="7.",description="load the origin7")
origin8 = ResponseSchema(name="8.",description="load the origin8")
origin9 = ResponseSchema(name="9.",description="load the origin9")
origin10 = ResponseSchema(name="10.",description="load the origin10")

change1 = ResponseSchema(name="Change1:",description="Change1:")
change2 = ResponseSchema(name="Change2:",description="Change2:")
change3 = ResponseSchema(name="Change3:",description="Change3:")
change4 = ResponseSchema(name="Change4:",description="Change4:")
change5 = ResponseSchema(name="Change5:",description="Change5:")
change6 = ResponseSchema(name="Change6:",description="Change6:")
change7 = ResponseSchema(name="Change7:",description="Change7:")
change8 = ResponseSchema(name="Change8:",description="Change8:")
change9 = ResponseSchema(name="Change9:",description="Change9:")
change10 = ResponseSchema(name="Change10:",description="Change10:")

judge1 = ResponseSchema(name="Check1:",description="Learn from these examples: if the sentence pair convey the same meaning, the check will be '0', otherwise '1'. Please adhere to this rule. ")
judge2 = ResponseSchema(name="Check2:",description="the same requirement as check1")
judge3 = ResponseSchema(name="Check3:",description="the same requirement as check1")
judge4 = ResponseSchema(name="Check4:",description="the same requirement as check1")
judge5 = ResponseSchema(name="Check5:",description="the same requirement as check1")
judge6 = ResponseSchema(name="Check6:",description="the same requirement as check1")
judge7 = ResponseSchema(name="Check7:",description="the same requirement as check1")
judge8 = ResponseSchema(name="Check8:",description="the same requirement as check1")
judge9 = ResponseSchema(name="Check9:",description="the same requirement as check1")
judge10 = ResponseSchema(name="Check10:",description="the same requirement as check1")

final_check_schemas = [origin1, change1, judge1, origin2, change2, judge2, origin3, change3, judge3,  origin4, change4, judge4, origin5, change5, judge5, origin6, change6, judge6, origin7, change7,\
     judge7, origin8, change8, judge8, origin9, change9, judge9, origin10, change10, judge10]
final_output_parser = StructuredOutputParser.from_response_schemas(final_check_schemas)
final_format_instructions = final_output_parser.get_format_instructions()



def retry_if_output_parser_exception(exception):

    return isinstance(exception, OutputParserException)

@retry(wait_random_min=256, wait_random_max=1000*10, stop_max_attempt_number=5,retry_on_exception=retry_if_output_parser_exception)
def write_check(llm, ans1_dict):
    conversation_write_check = ConversationChain(llm=llm,verbose=True)
    check_write_template = '''
        Role:
        <<<You are a demand checker. Learning from examples and check your cases.>>>

        Examples:
        <<< 
        1. "The dog chases the cat."
        Change: "The cat chases the dog."
        Check: 1 (They convey different meaning. The first one means the dog is the chaser, while the change means the cat is the chaser)

        2. "She heard he talked to the teacher"
        Change: "He heard she talked to the teacher" 
        Check: 1 (They convey different meaning. The first one means the girl is the listener and the boy is the talker, while the change means the boy is the chaser and the girl is the talker)

        3. "Exercise improves mental health."
        Change: "Mental health improves exercise."
        Check: 1 (They convey different meaning. The first one means exercise is the reason why mental health is improved, while the change means mental health is the reason why exercise is improved)

        4. "Yoga enhances physical flexibility."
        Change: "Physical flexibility is enhanced by yoga."
        Check: 0 (They convey the same meaning. Both mean yoga improves physical flexibility.)

        5. "Congress passed new legislation today."
        Change: "Today, new legislation was proposed in Congress."
        Check: 0 (They convey different meaning. Both mean Congress passed new legislation today)

        6. "A man rides a horse."
        Change: "A horse is ridden by a man."
        Check: 0 (They convey different meaning. Both mean the man rides a horse)
        >>>

        Your check cases:
        <<<
        1. {sent1}
        Change: {pair1}
        2. {sent2}
        Change: {pair2}
        3. {sent3}
        Change: {pair3}
        4. {sent4}
        Change: {pair4}
        5. {sent5}
        Change: {pair5}
        6. {sent6}
        Change: {pair6}
        7. {sent7}
        Change: {pair7}
        8. {sent8}
        Change: {pair8}
        9. {sent9}
        Change: {pair9}
        10. {sent10}
        Change: {pair10}
        >>>

        Demand: 
        <<<
        1. Learn from these examples: if the sentence pair convey the same meaning, the Check will be 0, otherwise 1. 
        2. Carefully analyze case by case, add notes. Don't be influenced by each other.
        3. Pay attention to the changes in relationships that cause shifts in meaning and the changes in voice that keep the meaning unchanged.
        >>> Please adhere to these rules.

        Your answer:
        <<<
        {format_instructions}
        >>>
    '''

    check_prompt = ChatPromptTemplate.from_template(template=check_write_template)
    prompt_w2 = check_prompt.format_messages(sent1 = ans1_dict['1.'],sent2 = ans1_dict['2.'],sent3 = ans1_dict['3.'],sent4 = ans1_dict['4.'],sent5 = ans1_dict['5.'],\
        sent6 = ans1_dict['6.'],sent7 = ans1_dict['7.'],sent8 = ans1_dict['8.'],sent9 = ans1_dict['9.'],sent10 = ans1_dict['10.'],\
        pair1 = ans1_dict['Change1:'], pair2 = ans1_dict['Change2:'], pair3 = ans1_dict['Change3:'], pair4 = ans1_dict['Change4:'], pair5 = ans1_dict['Change5:'],\
        pair6 = ans1_dict['Change6:'], pair7 = ans1_dict['Change7:'], pair8 = ans1_dict['Change8:'], pair9 = ans1_dict['Change9:'], pair10 = ans1_dict['Change10:'],\
        format_instructions=final_format_instructions)
    ans2 = conversation_write_check.predict(input = prompt_w2[0].content)
    ans2_dict = final_output_parser.parse(ans2)

    return ans2_dict
