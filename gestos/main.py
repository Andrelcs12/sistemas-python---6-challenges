
import json
import math
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PASTA = Path(__file__).resolve().parent
MODELO = PASTA / "hand_landmarker.task"
GRAVACOES = PASTA / "gravacoes"
DURACAO_REPLAY = 10
FPS_EXPORTACAO = 30
ANGULO_DEDOS = 155
ANGULO_POLEGAR = 150
LIMITE_PINCA = 0.32

CONEXOES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def distancia(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angulo(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    norma = math.hypot(*ab) * math.hypot(*cb)

    if norma == 0:
        return 0.0
    
    coseno = (ab[0] * cb[0] + ab[1] * cb[1]) / norma
    
    return math.degrees(math.acos(max(-1.0, min(1.0, coseno))))


def pixels_mao(mao, largura, altura):
    return [(x * largura, y * altura) for x, y, _ in mao]


def dedos_estendidos(mao, largura, altura):
    pontos = pixels_mao(mao, largura, altura)
    palma = max(distancia(pontos[5], pontos[17]), 1)
    polegar = (
        angulo(pontos[2], pontos[3], pontos[4]) >= ANGULO_POLEGAR
        and distancia(pontos[4], pontos[5]) >= palma * 0.75
    )
    dedos = [polegar]

    for base, meio, ponta in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        reto = angulo(pontos[base], pontos[meio], pontos[ponta]) >= ANGULO_DEDOS
        longe = distancia(pontos[ponta], pontos[0]) > distancia(pontos[meio], pontos[0]) * 1.12
        dedos.append(reto and longe)

    return dedos


def classificar_gesto(mao, largura, altura, dedos):
    pontos = pixels_mao(mao, largura, altura)
    palma = max(distancia(pontos[5], pontos[17]), 1)
    total = sum(dedos)

    if distancia(pontos[4], pontos[8]) / palma < LIMITE_PINCA:
        return "PINCA"
    if total == 0:
        return "PUNHO"
    if dedos[2] and total == 1:
        return "DEDO DO MEIO"
    if dedos[0] and dedos[1] and total == 2:
        return "FAZ O L"
    if total == 5:
        return "MAO ABERTA"
    if dedos[1] and dedos[2] and total == 2:
        return "PAZ"
    return "OUTRO"


def valor_estavel(amostras, minimo=3):
    if not amostras:
        return None
    valor, repeticoes = Counter(amostras).most_common(1)[0]
    return valor if repeticoes >= minimo else None


def centro_palma(mao):
    indices = (0, 5, 9, 13, 17)
    x = sum(mao[i][0] for i in indices) / len(indices)
    y = sum(mao[i][1] for i in indices) / len(indices)
    return x, y


def detectar_movimento(posicoes, agora, ultimo_evento):
    while posicoes and agora - posicoes[0][0] > 0.55:
        posicoes.popleft()
    if len(posicoes) < 3 or agora - ultimo_evento < 1.0:
        return None

    _, x0, y0 = posicoes[0]
    _, x1, y1 = posicoes[-1]
    dx, dy = x1 - x0, y1 - y0

    if abs(dx) > 0.22 and abs(dx) > abs(dy) * 1.6:
        posicoes.clear()
        return "DIREITA" if dx > 0 else "ESQUERDA"
    return None


def desenhar_maos(frame, maos):
    altura, largura = frame.shape[:2]
    for mao in maos:
        pontos = [(int(x * largura), int(y * altura)) for x, y, _ in mao]
        for a, b in CONEXOES:
            cv2.line(frame, pontos[a], pontos[b], (0, 220, 0), 2)
        for ponto in pontos:
            cv2.circle(frame, ponto, 4, (0, 0, 255), -1)


def escrever(frame, texto, y, cor=(255, 255, 255), tamanho=0.65):
    cv2.putText(frame, texto, (16, y), cv2.FONT_HERSHEY_SIMPLEX, tamanho, (0, 0, 0), 4)
    cv2.putText(frame, texto, (16, y), cv2.FONT_HERSHEY_SIMPLEX, tamanho, cor, 2)


def suavizar(anterior, atual):
    return atual if anterior == 0 else 0.8 * anterior + 0.2 * atual


def exportar_replay(dados, largura, altura):
    if not dados:
        return None
    GRAVACOES.mkdir(exist_ok=True)
    nome = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    arquivo_json = GRAVACOES / f"gestos_{nome}.json"
    arquivo_video = GRAVACOES / f"gestos_{nome}.mp4"
    inicio, fim = dados[0][0], dados[-1][0]

    registros = [
        {"tempo": round(t - inicio, 3), "maos": maos, "dedos": dedos}
        for t, maos, dedos in dados
    ]
    arquivo_json.write_text(
        json.dumps({"largura": largura, "altura": altura, "frames": registros}),
        encoding="utf-8",
    )

    video = cv2.VideoWriter(
        str(arquivo_video), cv2.VideoWriter_fourcc(*"mp4v"),
        FPS_EXPORTACAO, (largura, altura),
    )
    if not video.isOpened():
        video.release()
        arquivo_video = GRAVACOES / f"gestos_{nome}.avi"
        video = cv2.VideoWriter(
            str(arquivo_video), cv2.VideoWriter_fourcc(*"MJPG"),
            FPS_EXPORTACAO, (largura, altura),
        )
    if not video.isOpened():
        video.release()
        raise RuntimeError("Nao foi possivel criar o video")

    indice = 0
    quadros = max(1, int((fim - inicio) * FPS_EXPORTACAO) + 1)
    try:
        for numero in range(quadros):
            instante = inicio + numero / FPS_EXPORTACAO
            while indice + 1 < len(dados) and dados[indice + 1][0] <= instante:
                indice += 1
            _, maos, dedos = dados[indice]
            tela = np.zeros((altura, largura, 3), dtype=np.uint8)
            desenhar_maos(tela, maos)
            escrever(tela, f"Dedos: {dedos if dedos is not None else '--'}", 35)
            video.write(tela)
    finally:
        video.release()
    return arquivo_video


def executar():
    if not MODELO.is_file():
        raise FileNotFoundError(f"Modelo nao encontrado: {MODELO}")
    opcoes = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODELO)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Nao foi possivel abrir a webcam")

    historico = deque()
    contagens = deque(maxlen=5)
    gestos = deque(maxlen=5)
    posicoes = deque()
    esqueleto = True
    ultimo_evento = -10.0
    ultimo_movimento = ""
    movimento_ate = 0.0
    mensagem_replay = ""
    tarefa_replay = None
    fps_medio = inferencia_media = processamento_medio = 0.0
    anterior = None
    inicio = time.monotonic()
    ultimo_timestamp = -1

    try:
        with ThreadPoolExecutor(max_workers=1) as exportador:
            with mp.tasks.vision.HandLandmarker.create_from_options(opcoes) as detector:
                while True:
                    t0 = time.perf_counter()
                    sucesso, frame = camera.read()
                    if not sucesso:
                        break
                    frame = cv2.flip(frame, 1)
                    altura, largura = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    agora = time.monotonic()
                    timestamp = max(ultimo_timestamp + 1, int((agora - inicio) * 1000))
                    ultimo_timestamp = timestamp
                    t_ia = time.perf_counter()
                    resultado = detector.detect_for_video(imagem, timestamp)
                    inferencia_media = suavizar(inferencia_media, (time.perf_counter() - t_ia) * 1000)

                    maos = [[(p.x, p.y, p.z) for p in mao] for mao in resultado.hand_landmarks]
                    if maos:
                        dedos_por_mao = [dedos_estendidos(mao, largura, altura) for mao in maos]
                        contagem_bruta = sum(sum(dedos) for dedos in dedos_por_mao)
                        contagens.append(contagem_bruta)
                        contagem = valor_estavel(contagens)
                    else:
                        contagem_bruta = contagem = None
                        contagens.clear()

                    if len(maos) == 1:
                        classificacao = classificar_gesto(maos[0], largura, altura, dedos_por_mao[0])
                        gestos.append(classificacao)
                        gesto = valor_estavel(gestos)
                        x, y = centro_palma(maos[0])
                        posicoes.append((agora, x, y))
                        movimento = detectar_movimento(posicoes, agora, ultimo_evento)
                        if movimento:
                            ultimo_movimento = movimento
                            movimento_ate = agora + 1.0
                            ultimo_evento = agora
                    else:
                        gestos.clear()
                        posicoes.clear()
                        gesto = None

                    historico.append((agora, maos, contagem_bruta))
                    while historico and agora - historico[0][0] > DURACAO_REPLAY:
                        historico.popleft()

                    if tarefa_replay and tarefa_replay.done():
                        try:
                            caminho = tarefa_replay.result()
                            mensagem_replay = f"Salvo: {caminho.name}" if caminho else "Replay vazio"
                        except Exception as erro:
                            mensagem_replay = "Erro ao salvar replay"
                            print("Erro na exportacao:", erro)
                        tarefa_replay = None

                    if esqueleto:
                        desenhar_maos(frame, maos)
                    escrever(frame, f"Maos: {len(maos)} | Dedos: {contagem if contagem is not None else '--'}", 32)
                    escrever(frame, f"Gesto: {gesto or '--'}", 61)
                    if agora < movimento_ate:
                        escrever(frame, f"Movimento: {ultimo_movimento}", 90, (0, 255, 255))

                    processamento_media_atual = (time.perf_counter() - t0) * 1000
                    processamento_medio = suavizar(processamento_medio, processamento_media_atual)
                    if anterior is not None and agora > anterior:
                        fps_medio = suavizar(fps_medio, 1 / (agora - anterior))
                    anterior = agora
                    escrever(frame, f"FPS: {fps_medio:.1f} | IA: {inferencia_media:.1f}ms | Proc: {processamento_medio:.1f}ms", altura - 45, tamanho=0.48)
                    escrever(frame, "S: replay | K: pontos | Q: sair", altura - 18, tamanho=0.52)
                    if tarefa_replay:
                        escrever(frame, "Salvando replay...", altura - 72, (0, 255, 255), 0.55)
                    elif mensagem_replay:
                        escrever(frame, mensagem_replay[:65], altura - 72, (0, 255, 0), 0.52)

                    cv2.imshow("HandControl", frame)
                    tecla = cv2.waitKey(1) & 0xFF
                    if tecla == ord("q"):
                        break
                    if tecla == ord("k"):
                        esqueleto = not esqueleto
                    if tecla == ord("s") and tarefa_replay is None and historico:
                        tarefa_replay = exportador.submit(exportar_replay, list(historico), largura, altura)
                        mensagem_replay = ""
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    executar()
