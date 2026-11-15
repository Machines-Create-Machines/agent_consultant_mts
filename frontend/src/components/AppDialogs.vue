<script>
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
  methods: {
    authHeaders() {
      return {
        Authorization: this.token_type + " " + this.access_token,
        "Content-Type": "application/json",
      };
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
    async send() {
      const text = this.input.trim();
      if (!text || this.loading) return;
      
      this.loading = true;
      this.error = "";

      try {
        // Если диалог еще не создан в базе (activeDialogId === null), создаем его прямо сейчас
        if (!this.activeDialogId) {
          const res = await fetch(this.du + "/dialogs", {
            method: "POST",
            headers: this.authHeaders(),
          });
          if (!res.ok) throw new Error(await res.text());
          
          const dialog = await res.json();
          this.dialogs.unshift(dialog); // Добавляем в список слева
          this.activeDialogId = dialog.id; // Запоминаем ID нового диалога
        }

        // Теперь отправляем сообщение в уже существующий (или только что созданный) диалог
        const resMsg = await fetch(
          this.du + "/dialogs/" + this.activeDialogId + "/messages",
          {
            method: "POST",
            headers: this.authHeaders(),
            body: JSON.stringify({ content: text }),
          }
        );
        
        if (!resMsg.ok) throw new Error(await resMsg.text());
        
        const msg = await resMsg.json();
        this.messages.push(msg);
        this.input = ""; // Очищаем поле ввода
        
        // Обновляем список слева, чтобы подтянулся заголовок (title) диалога
        await this.loadDialogs();
        
        // Прокручиваем вниз
        this.$nextTick(() => {
          const box = this.$refs.msgbox;
          if (box) box.scrollTop = box.scrollHeight;
        });
      } catch (e) {
        this.error = "Не удалось создать диалог или отправить сообщение";
        console.error(e);
      } finally {
        this.loading = false;
      }
    }
    ,
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
      <ul>
        <li
          v-for="d in dialogs"
          :key="d.id"
          :class="{ active: d.id === activeDialogId }"
        >
          <span class="dialog-title" @click="openDialog(d.id)">
            {{ d.title }}
            <small>{{ d.messages_count }} сообщ.</small>
          </span>
          <button
            type="button"
            class="del-btn"
            title="Удалить диалог"
            @click="removeDialog(d.id)"
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
        <p v-if="!activeDialog" class="hint">
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
</style>
