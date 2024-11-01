import json
import duckdb
import pandas as pd
import openai

from pydantic import BaseModel, Field

conn = duckdb.connect(database=':memory:')


class AIFeedback(BaseModel):
    """A model representing an AI feedback response from the chatbot."""

    sql: str = Field(description="The SQL query string to be executed")
    intention: str = Field(
        description="The interpreted intention/purpose of the SQL query")
    need_sql: bool = Field(
        description="Whether the chatbot needs to execute a SQL query to answer the question")
    can_answer_by_sql: bool = Field(
        description="""Whether the chatbot can answer the question with executing a SQL query,
        if there is no related columns in the table, the chatbot should return False
        """)
    can_answer_by_prev_answers: bool = Field(
        description="Whether the chatbot can answer the question by using the previous answers")


class Memory(BaseModel):
    """A model representing a memory/message in the chatbot's conversation history.

    Attributes:
        role (str): Role of the message sender - must be 'user', 'ai/feedback','ai/answer' 'system', or 'duckdb'
        content (object): The content/payload of the memory/message
        status (str): Status of the memory - must be 'success', 'error', or 'unfinished'
    """
    role: str = Field(
        description="Role of the message, must be one of: 'user', 'ai/feedback','ai/answer', 'system', or 'duckdb'")
    content: object = Field(description="The content of the memory")
    status: str = Field(
        description="The status of the memory, must be one of: 'success', 'error', "
        "or 'unfinished'")


class ChatBot:
    """A chatbot that uses OpenAI to generate SQL queries and natural language responses.

    This class handles interactions with a DuckDB database through natural language by:
    1. Converting user questions into SQL queries using OpenAI
    2. Executing the queries against a DuckDB database
    3. Generating natural language responses from the query results

    Attributes:
        data_columns (list[str]): Schema information about the database columns
        data_sample (str): Sample data from the database for context
        data_path (str): Path to the JSON data file
        memory (list): History of interactions with the chatbot
        model (str): Name of the OpenAI model to use
        client (OpenAI): OpenAI client instance
    """

    data_columns: list[str] = []
    data_sample: str = None
    data_path: str = None
    df_json: pd.DataFrame = None

    sql_retries: int = 3
    previous_questions_in_context: int = 3

    def __init__(self, model: str, file_path: str):
        """Initialize the chatbot with a model and data file.

        Args:
            model (str): Name of the OpenAI model to use
            file_path (str): Path to the JSON data file to query against
        """
        self.data_path = file_path
        df = pd.read_json(file_path)

        # Code to generate data columns descriptions
        # Also need to convert the data type to json string
        # if it's a json column or list column
        # because duckdb doesn't support dict or list data type
        for column in df.columns:
            column_type = df[column].dtype
            description = f'Column {column} is of type {column_type}'
            if isinstance(df[column].values[0], dict):
                keys = pd.json_normalize(df[column]).columns.tolist()
                description = f'Column {column} is a json column with keys: {keys}'
                df[column] = df[column].apply(json.dumps)

            if isinstance(df[column].values[0], list):
                description = f'Column {column} is a list column'
                df[column] = df[column].apply(json.dumps)
            self.data_columns += f"{column}: {description}\n"

        self.df_json = df

        self.data_sample = str(df.head())
        self.memory = []
        self.model = model
        self.client = openai.OpenAI()

    def get_previous_questions(self) -> str:
        """Get the most recent user questions from memory.

        Returns:
            str: String containing the last N user questions, joined by newlines,
                where N is determined by previous_questions_in_context.
        """
        previous_questions = [x.content['user_message']
                              for x in self.memory if x.role == 'user']
        return '\n'.join(previous_questions[-self.previous_questions_in_context:])

    def get_previous_answers(self) -> str:
        """Get the most recent AI answers from memory.

        Returns:
            str: String containing the last N AI answers, joined by newlines,
                where N is determined by previous_questions_in_context.
        """
        previous_answers = [x.content['answer']
                            for x in self.memory if x.role in ['ai/answer', 'system']]
        return '\n'.join(previous_answers[-self.previous_questions_in_context:])

    def get_sql_from_ai(self) -> AIFeedback:
        """Get an SQL query from the AI model based on the user's message.

        This method sends a prompt to the OpenAI model asking it to generate a SQL query
        based on the user's message. The prompt includes the database schema and sample 
        data to help the model generate an appropriate query.

        Returns:
            list: The results from executing the generated SQL query

        Raises:
            Exception: If there is an error executing the SQL query
        """
        template = """
        You are a helpful assistant that helps users query a duckdb database.
        You are given a message from a user and you need to return a valid SQL query.
        We are using duckdb, so your SQL query should be valid for duckdb.
        The table name is "{data_path}", and it is a json file.
        The table schema is:
        {data_schema}
        The sample data is:
        {data_sample}
        The user's message is:
        {user_message}
        
        Please don't include julian date in the SQL query, 
        because duckdb doesn't support julian date.
        
        Please notice that some columns may have null values,
        when you group by a column, you need to filter out the null values.
        you need to cast it to basic type like double, int, string, etc.
        
        Let's think step by step.
        """
        current_user_message = list(filter(lambda x: x.role == 'user',
                                           self.memory))[-1].content[
            'user_message']
        # ugly code, but it's required if we use duckdb
        df_json = self.df_json
        prompt = template.format(
            data_path='df_json', data_schema=self.data_columns,
            data_sample=self.data_sample, user_message=current_user_message)

        template_with_prev_template = """
        We got a sql from you before, but it failed to execute.
        Please fix the sql query and return a new one.
        ========================================================
        Previous SQL query:
        {prev_sql}
        Previous SQL query error message:
        {prev_sql_error_message}
        ========================================================
        """

        if self.memory[-1].role == 'duckdb' and self.memory[-1].status == 'error':
            prompt += template_with_prev_template.format(
                prev_sql=self.memory[-1].content['sql'],
                prev_sql_error_message=self.memory[-1].content['error'])

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=AIFeedback,
        )
        # return response.choices[0].message.parsed
        self.memory.append(
            Memory(role='ai/feedback', content=response.choices[0].message.parsed,
                   status='success'))

        if response.choices[0].message.parsed.need_sql:
            try:
                sql_results = conn.sql(
                    response.choices[0].message.parsed.sql).fetchall()
                self.memory.append(
                    Memory(role='duckdb', content={
                        'sql_results': sql_results,
                        'sql': response.choices[0].message.parsed.sql
                    }, status='success'))

            except Exception as e:
                self.memory.append(
                    Memory(role='duckdb', content={
                        'sql': response.choices[0].message.parsed.sql,
                        'error': str(e)
                    }, status='error'))

    def generate_response_from_ai(self, user_message: str) -> str:
        """Generate a natural language response from the AI model based on SQL results.

        This method:
        1. Adds the user message to memory
        2. Gets SQL query from AI and executes it (with retries)
        3. Generates natural language response from results
        4. Updates memory with the response

        Args:
            user_message (str): The user's natural language query message

        Returns:
            str: Natural language response explaining the query results
            None: If there was an error executing the query
        """
        template = """
        You are a helpful assistant that helps users understand the results of a SQL query.
        The SQL query results are:
        {sql_results}
        The user's message is:
        {user_message}
        Please respond to the user's message based on the SQL query results.
        
        ========================================================
        This user may have asked a question before, please use the previous 
        answer to help you answer the current question.
        Previous questions:
        {previous_questions}
        
        Previous answers:
        {previous_answers}
        ========================================================
        
        Please respond directly to the user's message,
        don't mention the result is from a SQL query.
        
        Please don't include the process of getting the result,
        instead, just provide concise and accurate information.
        """
        # Simple Retry
        self.memory.append(Memory(role='user', content={
            'user_message': user_message
        }, status='unfinished'))
        for _ in range(self.sql_retries):
            self.get_sql_from_ai()
            if self.memory[-1].status == 'success':
                break

        # we hide the error and return a default error message
        if self.memory[-1].status == 'error':
            self.memory.append(Memory(role='system', content={
                'error': 'I am sorry, I cannot answer your question.'
            }, status='error'))
            return self.memory[-1].content['error']

        # Generate response from AI
        ai_feedback = list(
            filter(lambda x: x.role == 'ai/feedback', self.memory))[-1].content
        # Some tricky questions cannot be answered by SQL or previous answers
        # Questions have no related columns in the table
        if not ai_feedback.can_answer_by_sql and not ai_feedback.can_answer_by_prev_answers:
            self.memory.append(Memory(role='system', content={
                'error': 'I am sorry, I cannot answer your question.'
            }, status='error'))
            return self.memory[-1].content['error']

        sql_results = ''
        if ai_feedback.need_sql:
            sql_results = self.memory[-1].content['sql_results']

        prompt = template.format(
            sql_results=sql_results, user_message=user_message,
            previous_questions=self.get_previous_questions(),
            previous_answers=self.get_previous_answers())

        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],

        )
        answer = response.choices[0].message.content
        self.memory.append(
            Memory(role='ai/answer', content={
                'answer': answer
            }, status='success'))
        user_message = list(
            filter(lambda x: x.role == 'user', self.memory))[-1]
        user_message.content['answer'] = answer
        user_message.status = 'success'
        return answer
