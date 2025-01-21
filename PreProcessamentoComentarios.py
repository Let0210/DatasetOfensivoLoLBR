import pandas as pd
import emoji

dicionario = {
    # Gírias e abreviações gerais
    "vc": "você",
    "vcs": "vocês",
    "tb": "também",
    "blz": "beleza",
    "mt": "muito",
    "pq": "por que",
    "q": "que",
    "td": "tudo",
    "n": "não",
    "s": "sim",
    "eh": "é",
    "pra": "para",
    "tá": "está",
    "ta": "está",
    "sao": "são",
    "tbm": "também",
    "vlw": "valeu",
    "dps": "depois",
    "to": "estou",
    "c": "com",
    "cmg": "comigo",
    "mto": "muito",
    "aki": "aqui",
    "pf": "por favor",
    "pvc": "por você",
    "hj": "hoje",
    "tmj": "tamo junto",
    "kkkk": "ri demais",
    "krl": "caralho",
    "top": "excelente",
    "ggz": "jogo tranquilo",
    "troll": "prejudicar intencionalmente",
    "imba": "desbalanceado",
    "nerfado": "enfraquecido",
    "ez": "fácil",
    "wtf": "que droga",
    "omg": "meu deus",
    "pls": "por favor",
    "stfu": "cale a boca",
    "afk": "ausente",
    "brb": "já volto",
    "ty": "obrigado",
    "np": "sem problema",
    "fdp": "filho da puta",
    "fdps":"filhos da puta",

    # Gírias de jogos online
    "gg": "bom jogo",
    "noob": "iniciante",
    "carry": "carregar",
    "nerf": "enfraquecer",
    "buff": "fortalecer",
    "lag": "lentidão",
    "fps": "frames por segundo",
    "br": "brasil",
    "skin": "aparência",
    "skill": "habilidade",
    "rush": "atacar rapidamente",
    "farm": "coletar recursos",
    "ping": "tempo de resposta",
    "ult": "habilidade final",
    "fair play": "jogo justo",
    "flw": "até mais",
    "partiu": "vamos",
    "zuera": "brincadeira",
    "gnt": "gente",
    "mana": "moça",
    "mano": "rapaz",
    "x1": "um contra um",
    "irl": "na vida real",
    "nt": "boa tentativa",
    "bait": "isca",
    "camper": "jogador que se esconde",

    # Específicas de lol
    "all-in": "vai com tudo pra cima do adversário",
    "backdoor": "atacar uma construção sem creeps",
    "brush": "moita",
    "bush": "moita",
    "burst damage": "bastante dano em pouco tempo",
    "champion": "personagem do jogo",
    "chase": "perseguir",
    "coach": "treinador",
    "core build": "itens principais de um personagem",
    "dash": "movimento rápido",
    "desengage": "evitar uma luta",
    "dps": "dano por segundo",
    "engage": "forçar uma luta",
    "facecheck": "checar uma área sem saber se é segura",
    "feedar": "morrer várias vezes",
    "fp": "jogo justo",
    "gapclose": "diminuir distância",
    "givar": "desistir",
    "hitbox": "área sujeita a dano",
    "kite": "andar e bater",
    "jungle": "selva do mapa",
    "lane": "rota do mapa",
    "mid": "rota do meio",
    "top": "rota do topo",
    "adc": "atirador",
    "support": "suporte",
    "midlaner": "jogador da rota do meio",
    "toplaner": "jogador da rota do topo",
    "jungler": "jogador da selva",
    "gank": "surpreender o adversário em uma rota",
    "splitpush": "pressionar rotas separadamente",
    "cc": "controle de grupo",
    "aoe": "dano em área",
    "ward": "item de visão",
    "vision": "controle de visão no mapa",
    "objective": "alvo estratégico, como dragão ou barão",
    "baron": "Barão Nashor, personagem épico",
    "herald": "Arauto do Vale, personagem épico",
    "dragon": "Dragão Elemental",
    "elder": "Dragão Ancião",
    "siege": "pressionar torres",
    "snowball": "crescer rapidamente no jogo",
    "roam": "deixar sua rota para ajudar outra",
    "poke": "atacar de longe para enfraquecer o inimigo",
    "zoning": "forçar o inimigo a recuar de uma área",
    "power spike": "pico de poder em determinado momento",
    "reset": "voltar à base para se curar e comprar itens",
    "tank": "personagem que absorve muito dano",
    "bruiser": "personagem resistente com dano moderado",
    "mage": "personagem com foco em dano mágico",
    "assassin": "personagem com alto dano explosivo",
    "farmar": "coletar recursos",
}



def remover_emojis(texto):
    return emoji.replace_emoji(texto, "")

def limpar_texto(texto, dicionario):
    palavras = texto.split()
    palavras_norm = [dicionario.get(p, p) for p in palavras]
    texto = " ".join(palavras_norm)
    
    texto = emoji.replace_emoji(texto, "")

    return texto

   
def minusculo(texto):
    return texto.lower()


arq = 'comentariosAntes.csv'
arq_preProcessado = "comentariosAClassificar.csv"

df = pd.read_csv(arq)


df['message'] = df['message'].apply(limpar_texto, args=(dicionario,))

df = df[df['message'].str.strip() != ""]  # Remove linhas de strings vazias

df.to_csv(arq_preProcessado, index=False)


