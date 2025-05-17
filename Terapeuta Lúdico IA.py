%pip install google-genai
import os
from google.colab import userdata
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')
from google import genai

client = genai.Client()
modelo = "gemini-1.5-pro-latest"
%pip install -q google-adk
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types  # Para criar conteúdos (Content e Part)
from datetime import date
import textwrap # Para formatar melhor a saída de texto
from IPython.display import display, Markdown # Para exibir texto formatado no Colab
import requests # Para fazer requisições HTTP
import warnings

warnings.filterwarnings("ignore")
# Função auxiliar que envia uma mensagem para um agente via Runner e retorna a resposta final
def call_agent(agent: Agent, message_text: str) -> str:
    # Cria um serviço de sessão em memória
    session_service = InMemorySessionService()
    # Cria uma nova sessão (você pode personalizar os IDs conforme necessário)
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    # Cria um Runner para o agente
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    # Cria o conteúdo da mensagem de entrada
    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response = ""
    # Itera assincronamente pelos eventos retornados durante a execução do agente
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
          for part in event.content.parts:
            if part.text is not None:
              final_response += part.text
              final_response += "\n"
    return final_response
    # Função auxiliar para exibir texto formatado em Markdown no Colab
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
  # Agente de brainstorm

def agente_brainstormer(objetivo, idade, interesse):
    brainstormer = Agent(
        name="Brainstormer",
        model = "gemini-1.5-pro-latest",
        instruction ="""
        Você terá a responsabilidade de receber a o obetivo terapêutico(o que o fonoaudiólogo quer produzir
        para ajudar a fala da criança), a idade da criança e os interesses da mesma. Com estas informações
        você deve fazer uma pesquisa no Google com a ferramenta {google_search} sobre atividades que possam
        ajudar o profisional a ajudar a criança mantendo seu interesse, achando atividades que devem
        ser filtradas por relevância, um artivo pouco relevante sobre o assunto pode ser ineficáz e apenas
        atrapalhará a pesquisa.
        """,
        description = "Agente que busca as ideias de atividades",
        tools = [google_search]
    )
    saida_brainstormer = f"Objetivo terapêutico: {objetivo}\nIdade: {idade}\nInteresses: {interesse}"
    ideias_geradas = call_agent(brainstormer, saida_brainstormer)
    return ideias_geradas
    # Agente de análise

def agente_curator(objetivo, idade, interesse, ideias_geradas):
    curator = Agent(
        name ="Curator",
        model = "gemini-1.5-pro-latest",
        instruction ="""
        Você terá a responsabilidade de analisar a lista de ideias geradas pelo "Brainstormer", avaliando a
        relevância da atividade de acordo com o objetivo desejado, a idade da criança e os interesses da 
        mesma. Certificando que as ideias selecionadas são adequadas para a idade da criança e promissoras.
        Deve organiza-las, dando informações importantes sobre a atividade para que esta possa ser 
        posteriormentedetalhadas e apresentadas.
        """,
        description = "Agente que organiza as ideias de atividades",
        tools = [google_search]
    )
    saida_curator = f"Objetivo terapêutico: {objetivo}\nIdade: {idade}\nInteresses: {interesse}\nIdeias geradas: {ideias_geradas}"
    ideias_organizadas = call_agent(curator, saida_curator)
    return ideias_organizadas
    # Agente de planejamento

def agente_planner(objetivo, idade, interesse, ideias_organizadas):
    planner = Agent(
        name ="Planner",
        model = "gemini-1.5-pro-latest",
        instruction ="""
        Você terá a responsabilidade de pegar as ideias organizadas pelo "curator" e a partir delas criar
        um plano de atividades detalhado. Incluíndo material utilizado, tempo médio da atividade, passo a
        passo para como o fonoaudiólogo pode desenvolver e "jogar" a atividade para ajudar a criança.
        """,
        description = "Agente que detalha as atividades em um plano prático"
    )
    saida_planner = f"Objetivo terapêutico: {objetivo}\nIdade: {idade}\nInteresses: {interesse}\nIdeias organizadas: {ideias_organizadas}"
    plano_de_atividades = call_agent(planner, saida_planner)
    return plano_de_atividades
    # Agente de revisão

def agente_reviewer(objetivo, idade, interesse, ideias_organizadas):
    reviewer = Agent(
        name ="Reviewer",
        model = "gemini-1.5-pro-latest",
        instruction ="""
        Você terá a responsabilidade de analisar o plano de atividades gerado pelo "Planner" para garantir
        que ele esteja no tom adequado para um profissional de fonoaudiologia, que a linguagem seja clara e
        profissional, e que a atividade seja eficaz para ajudar a criança na situação problema, corrigindo
        erros ortográficos e gramáticos. O "Reviewer" também pode identificar possíveis melhorias ou pontos
        a serem clarificados.
        """,
        description = "Agente que analiza a atividade gerada pelo plannar e corrige possíveis erros"
    )
    saida_reviewer = f"Objetivo terapêutico: {objetivo}\nIdade: {idade}\nInteresses: {interesse}\nIdeias organizadas: {ideias_organizadas}"
    plano_revisado = call_agent(reviewer, saida_reviewer)
    return plano_revisado
    print("🚀 Iniciando o Sistema de Terapeuta Lúdico IA com 4 Agentes 🚀")
# --- Obter o Tópico do Usuário ---
objetivo = input("Digite o objetivo da atividade: ")
idade = input("Digite a idade da criança: ")
interesse = input("Digite os interesses da criança: ")


# Inserir lógica do sistema de agentes ################################################
if not objetivo:
  print("Por favor, insira um tópico válido.")
else:
  print(f"Vamos trabalhar sobre este tópico: {objetivo} ")

  atividades = agente_brainstormer(objetivo, idade, interesse)
  print("resultado do agente buscador")
  display(to_markdown(atividades))
  print("------------------------------")