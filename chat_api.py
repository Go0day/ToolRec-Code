import openai
import argparse



def llm_chat(User_message, stop='12'):
    openai.api_key = "sk-XXXX"
    openai.api_base = "XXXX"

    if len(stop) < 3:
        stop=None
    else:
        stop = [stop]
    our_messages = [
        {'role': 'user', 'content': User_message}
    ]
    if stop:
        response = openai.ChatCompletion.create(
            #model='gpt-3.5-turbo',
            # model='gpt-3.5-turbo-16k',
            model='gpt-4o', # test
            messages=our_messages,
            stop=stop
        )
    else:
        response = openai.ChatCompletion.create(
            #model='gpt-3.5-turbo',
            # model='gpt-3.5-turbo-16k',
            model='gpt-4o', # test
            messages=our_messages
        )
    llm_response =  response['choices'][0]['message'].to_dict()['content']
    #return r"{0}".format(repr(llm_response)[1:-1])
    return llm_response


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--role", "-r", type=str, default="Movie recommender", help="llm role")
#     parser.add_argument("--message", "-m", type=str, default="Please recommend some movie for user", help="Messages send to chatgpt.")
#     parser.add_argument("--stop", "-s", type=str, default="1", help="Stop Words")

#     args, _ = parser.parse_known_args()
#     print(llm_chat(args.role + args.message, stop=args.stop))
#     # print(llm_chat_renmin(args.message))