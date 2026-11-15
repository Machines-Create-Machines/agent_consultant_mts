<script>
const WS_RECONNECT_BASE_MS = 1000;
const WS_RECONNECT_MAX_MS = 30000;
const WS_PING_INTERVAL_MS = 25000;

export default {
  name: "AppDialogs",
  props: {
    du: { type: String, required: true },
    token_type: { type: String, required: true },
    access_token: { type: String, required: true },
  },
  data() {
    return {
      dialogs: [],
      activeDialogId: null,
      messages: [],
      input: "",
      loading: false,
      error: "",
      ws: null,
      wsReady: false,
      wsStatus: "connecting", // connecting | online | offline
      reconnectAttempt: 0,
      reconnectTimer: null,
      pingTimer: null,
      intentionalClose: false,
    };
  },
  computed: {
    activeDialog() {
      return this.dialogs.find((d) => d.id === this.activeDialogId) || null;
    },
  },
  created() {
    this.loadDialogs();
  },
  mounted() {
    this.connectWs();
  },
  beforeUnmount() {
    this.closeWs();
  },
  methods: {
    authHeaders() {
      return {
        Authorization: this.token_type + " " + this.access_token,
        "Content-Type": "application/json",
      };
    },
    wsChatUrl() {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      const token = encodeURIComponent(this.access_token);
      return `${proto}//${window.location.host}/ws/chat?token=${token}`;
    },
    connectWs() {
      if (!this.access_token) return;
      this.clearReconnectTimer();
      this.clearPingTimer();
      this.intentionalClose = false;

      if (this.ws) {
        try {
          this.ws.onopen = null;
          this.ws.onmessage = null;
          this.ws.onerror = null;
          this.ws.onclose = null;
          this.ws.close();
        } catch (_) {
          /* ignore */
        }
        this.ws = null;
      }

      this.wsStatus = "connecting";
      const socket = new WebSocket(this.wsChatUrl());
      this.ws = socket;
      this.wsReady = false;

      socket.onopen = () => {
        this.reconnectAttempt = 0;
        this.startPing();
      };

      socket.onmessage = (ev) => {
        this.handleWsMessage(ev);
      };

      socket.onerror = () => {
        // onclose handles reconnect; keep UI quiet unless send fails
      };

      socket.onclose = () => {
        this.wsReady = false;
        this.wsStatus = "offline";
        this.loading = false;
        this.clearPingTimer();
        if (this.ws === socket) this.ws = null;
        if (!this.intentionalClose) this.scheduleReconnect();
      };
    },
    closeWs() {
      this.intentionalClose = true;
      this.clearReconnectTimer();
      this.clearPingTimer();
      this.wsReady = false;
      this.wsStatus = "offline";
      this.loading = false;
      if (this.ws) {
        try {
          this.ws.close();
        } catch (_) {
          /* ignore */
        }
        this.ws = null;
      }
    },
    scheduleReconnect() {
      this.clearReconnectTimer();
      this.wsStatus = "connecting";
      const delay = Math.min(
        WS_RECONNECT_BASE_MS * Math.pow(2, this.reconnectAttempt),
        WS_RECONNECT_MAX_MS
      );
      this.reconnectAttempt += 1;
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        this.connectWs();
      }, delay);
    },
    clearReconnectTimer() {
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
    },
    startPing() {
      this.clearPingTimer();
      this.pingTimer = setInterval(() => {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ type: "ping" }));
        }
      }, WS_PING_INTERVAL_MS);
    },
    clearPingTimer() {
      if (this.pingTimer) {
        clearInterval(this.pingTimer);
        this.pingTimer = null;
      }
    },
    handleWsMessage(ev) {
      let data;
      try {
        data = JSON.parse(ev.data);
      } catch (_) {
        return;
      }
      const type = data && data.type;
      if (!type) return;

      switch (type) {
        case "ready":
          this.wsReady = true;
          this.wsStatus = "online";
          this.error = "";
          break;
        case "dialog": {
          const dialog = this.normalizeDialogFrame(data);
          if (dialog && dialog.id) {
            this.activeDialogId = dialog.id;
            const idx = this.dialogs.findIndex((d) => d.id === dialog.id);
            if (idx >= 0) {
              this.dialogs.splice(idx, 1, { ...this.dialogs[idx], ...dialog });
            } else {
              this.dialogs.unshift(dialog);
            }
          }
          break;
        }
        case "user_message": {
          const msg = this.normalizeMessageFrame(data);
          if (msg && msg.id != null) {
            if (!this.messages.some((m) => m.id === msg.id)) {
              this.messages.push(msg);
            }
            this.scrollMessages();
          }
          break;
        }
        case "assistant_message": {
          const msg = this.normalizeMessageFrame(data);
          if (msg && msg.id != null) {
            if (!this.messages.some((m) => m.id === msg.id)) {
              this.messages.push(msg);
            }
            this.scrollMessages();
            this.loadDialogs();
          }
          break;
        }
        case "status": {
          const state = data.state;
          if (state === "thinking") {
            this.loading = true;
          } else if (state === "done") {
            this.loading = false;
          }
          break;
        }
        case "error":
          this.error = (data.detail && String(data.detail)) || "Ошибка чата";
          this.loading = false;
          break;
        case "pong":
          break;
        default:
          break;
      }
    },
    normalizeDialogFrame(data) {
      const src = data.dialog && typeof data.dialog === "object" ? data.dialog : data;
      if (!src || src.id == null) return null;
      return {
        id: src.id,
        title: src.title,
        created_at: src.created_at,
        messages_count: src.messages_count,
        updated_at: src.updated_at,
      };
    },
    normalizeMessageFrame(data) {
      const src = data.message && typeof data.message === "object" ? data.message : data;
      if (!src || src.id == null) return null;
      return {
        id: src.id,
        role: src.role,
        content: src.content,
        documents: src.documents || null,
        created_at: src.created_at,
        dialog_id: src.dialog_id,
      };
    },
    scrollMessages() {
      this.$nextTick(() => {
        const box = this.$refs.msgbox;
        if (box) box.scrollTop = box.scrollHeight;
      });
    },
    fmtTime(iso) {
      if (!iso) return "";
      // Добавляем 'Z', если бэкенд присылает время без указания зоны,
      // чтобы браузер понял, что это UTC и перевел в локальное время.
      let dateStr = iso;
      if (!dateStr.endsWith("Z") && !dateStr.includes("+")) {
        dateStr += "Z";
      }
      const d = new Date(dateStr);
      return d.toLocaleString("ru-RU", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    },
    async loadDialogs() {
      this.error = "";
      try {
        const res = await fetch(this.du + "/dialogs", {
          headers: this.authHeaders(),
        });
        if (!res.ok) throw new Error(await res.text());
        this.dialogs = await res.json();
        if (
          this.activeDialogId &&
          !this.dialogs.some((d) => d.id === this.activeDialogId)
        ) {
          this.activeDialogId = null;
          this.messages = [];
        }
      } catch (e) {
        this.error = "Не удалось загрузить список диалогов";
      }
    },
    async newDialog() {
      // Просто сбрасываем активный диалог. В базу ничего не пишем!
      this.activeDialogId = null;
      this.messages = [];
      this.input = "";
      this.error = "";
    },
    async openDialog(id) {
      this.activeDialogId = id;
      this.messages = [];
      this.error = "";
      try {
        const res = await fetch(this.du + "/dialogs/" + id + "/messages", {
          headers: this.authHeaders(),
        });
        if (!res.ok) throw new Error(await res.text());
        this.messages = await res.json();
        this.scrollMessages();
      } catch (e) {
        this.error = "Не удалось загрузить сообщения";
      }
    },
    async removeDialog(id) {
      if (!confirm("Удалить диалог?")) return;
      try {
        const res = await fetch(this.du + "/dialogs/" + id, {
          method: "DELETE",
          headers: this.authHeaders(),
        });
        if (!res.ok) throw new Error(await res.text());
        if (this.activeDialogId === id) {
          this.activeDialogId = null;
          this.messages = [];
        }
        await this.loadDialogs();
      } catch (e) {
        this.error = "Не удалось удалить диалог";
      }
    },
    send() {
      const text = this.input.trim();
      if (!text || this.loading) return;

      if (!this.ws || this.ws.readyState !== WebSocket.OPEN || !this.wsReady) {
        this.error = "Нет соединения с чатом. Подождите или обновите страницу.";
        return;
      }

      this.error = "";
      this.input = "";

      this.ws.send(
        JSON.stringify({
          type: "user_message",
          dialog_id: this.activeDialogId,
          content: text,
        })
      );
    },
    docTitles(documents) {
      if (!documents || !documents.length) return [];
      return documents.map((d) => d.title || d.id_cr || JSON.stringify(d));
    },
  },
};
</script>

<template>
  <div class="dialogs">
    <aside class="dialog-list">
      <button type="button" class="new-dialog-btn" @click="newDialog">
        Начать новый диалог
      </button>
      <p class="ws-status" :class="wsStatus">
        {{
          wsStatus === "online"
            ? "Чат подключён"
            : wsStatus === "connecting"
              ? "Подключение к чату…"
              : "Нет связи с чатом, переподключение…"
        }}
      </p>
      <ul>
        <li
          v-for="d in dialogs"
          :key="d.id"
          :class="{ active: d.id === activeDialogId }"
          @click="openDialog(d.id)"
        >
          <span class="dialog-title">
            {{ d.title }}
            <small>{{ d.messages_count }} сообщ.</small>
          </span>
          <button
            type="button"
            class="del-btn"
            title="Удалить диалог"
            @click.stop="removeDialog(d.id)"
          >
            ✕
          </button>
        </li>
      </ul>
      <p v-if="!dialogs.length" class="hint">
        Диалогов пока нет. Нажмите «Начать новый диалог».
      </p>
    </aside>

    <section class="dialog-window">
      <header v-if="activeDialog">
        <strong>{{ activeDialog.title }}</strong>
        <span class="created">Создан: {{ fmtTime(activeDialog.created_at) }}</span>
      </header>
      <div ref="msgbox" class="messages">
        <p v-if="!activeDialog && !messages.length" class="hint">
          Выберите диалог слева или начните новый.
        </p>
        <div
          v-for="m in messages"
          :key="m.id"
          class="message"
          :class="m.role"
        >
          <div class="meta">{{ fmtTime(m.created_at) }}</div>
          <div class="content">{{ m.content }}</div>
          <div v-if="docTitles(m.documents).length" class="docs">
            Использованные документы:
            <ul>
              <li v-for="(t, i) in docTitles(m.documents)" :key="i">{{ t }}</li>
            </ul>
          </div>
        </div>
        <p v-if="loading" class="thinking">Ассистент думает…</p>
      </div>
      <div class="input-row">
        <textarea
          v-model="input"
          rows="2"
          placeholder="Задайте вопрос..."
          @keydown.enter.exact.prevent="send"
        ></textarea>
        <button type="button" :disabled="loading || !input.trim()" @click="send">
          Отправить
        </button>
      </div>
      <p v-if="error" class="error">{{ error }}</p>
    </section>
  </div>
</template>

<style scoped>
.dialogs {
  display: flex;
  gap: 12px;
  padding: 12px;
}
.dialog-list {
  width: 240px;
  flex-shrink: 0;
  border-right: 1px solid #ddd;
  padding-right: 12px;
}
.new-dialog-btn {
  width: 100%;
  margin-bottom: 8px;
  padding: 8px;
  cursor: pointer;
}
.dialog-list ul {
  list-style: none;
  margin: 0;
  padding: 0;
}
.dialog-list li {
  display: flex;
  align-items: center; /* Выравниваем по центру по вертикали */
  justify-content: space-between; /* Текст слева, крестик справа */
  padding: 12px; 
  margin-bottom: 8px;
  border-radius: 12px;
  transition: background 0.2s;
  background: white; /* Чтобы карточки выделялись на сером фоне */
  cursor: pointer;
}
.dialog-list li.active {
  background: #e8f0fe;
  border-left: 4px solid #E30611;
}
.dialog-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1; /* Занимает всё доступное место */
  font-weight: 500;
}
.dialog-title small {
  color: #888;
  margin-left: 4px;
  font-size: 0.8em;
}
.del-btn {
  border: none;
  background: transparent;
  color: #E30611;
  cursor: pointer;
  font-size: 14px;
  font-weight: bold;
  
  /* Жестко задаем размеры, чтобы кнопка была идеальным кругом */
  width: 28px;
  height: 28px;
  min-width: 28px; /* Запрещаем сжиматься */
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%; /* Теперь это идеальный круг */
  
  transition: background 0.2s, color 0.2s;
  margin-left: 10px; /* Небольшой отступ от текста */
}
.del-btn:hover {
  background: #FFE8E8; /* Легкая красная подсветка при наведении */
  color: white;
}
.dialog-window {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 70vh;
}
.dialog-window header {
  display: flex;
  justify-content: space-between;
  padding: 8px;
  border-bottom: 1px solid #ddd;
}
.created {
  color: #888;
  font-size: 0.85em;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.message {
  max-width: 75%;
  padding: 8px 10px;
  border-radius: 8px;
  background: #f1f3f4;
  align-self: flex-start;
}
.message.user {
  align-self: flex-end;
  background: #d2e3fc;
}
.message .meta {
  font-size: 0.75em;
  color: #777;
  margin-bottom: 4px;
}
.message .docs {
  margin-top: 6px;
  font-size: 0.85em;
  color: #555;
  border-top: 1px dashed #bbb;
  padding-top: 4px;
}
.message .docs ul {
  margin: 2px 0 0;
  padding-left: 18px;
}
.thinking {
  align-self: flex-start;
  color: #666;
  font-style: italic;
  margin: 4px 0;
}
.input-row {
  display: flex;
  gap: 8px;
  padding: 8px;
  border-top: 1px solid #ddd;
}
.input-row textarea {
  flex: 1;
  resize: vertical;
}
.hint {
  text-align: center; /* Центрируем текст */
  margin: auto; /* Центрируем сам блок на экране */
  font-size: 20px; /* Увеличиваем шрифт */
  font-weight: 600; /* Делаем жирным */
  color: #1D1D1B; /* Темный цвет МТС */
  max-width: 80%;
  line-height: 1.5;
}
.error {
  color: #c00;
  margin: 0 8px;
}
.ws-status {
  font-size: 0.8em;
  margin: 0 0 8px;
  padding: 4px 8px;
  border-radius: 6px;
}
.ws-status.online {
  color: #1b7a3d;
  background: #e8f7ee;
}
.ws-status.connecting {
  color: #8a6d00;
  background: #fff6d6;
}
.ws-status.offline {
  color: #a11;
  background: #ffe8e8;
}
</style>
