<script>
export default {
  name: "LeaderBoard",
  props: {
    du: { type: String, required: true },
    role: String,
    token_type: String,
    access_token: String,
    login: String,
  },
  data() {
    return {
      chlogin: "",
      chfname: "",
      chlname: "",
      chrole: "",
      chcrat: "",
      redact: false,
      error: "",
    };
  },
  created() {
    if (this.login) {
      this.opprof(this.login);
    }
  },
  methods: {
    async opprof(login) {
      this.error = "";
      try {
        const response = await fetch(this.du + "/profile?login=" + login);
        if (!response.ok) {
          throw new Error(`Response status: ${response.status}`);
        }
        const data = await response.json();
        const usr = data[0];
        this.chlogin = usr.login;
        this.chfname = usr.first_name;
        this.chlname = usr.last_name;
        this.chrole = usr.role;
        this.chcrat = usr.created_at;
      } catch (error) {
        this.error = "Не удалось загрузить профиль";
      }
    },
    fmtTime(iso) {
      if (!iso) return "";
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
    toredact() {
      this.redact = true;
    },
    outredact() {
      this.redact = false;
      this.opprof(this.login);
    },
    async change() {
      this.error = "";
      try {
        const response = await fetch(this.du + "/profile", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: this.token_type + " " + this.access_token,
          },
          body: JSON.stringify({
            first_name: this.chfname,
            last_name: this.chlname,
          }),
        });
        if (!response.ok) {
          throw new Error(`Response status: ${response.status}`);
        }
        this.outredact();
      } catch (error) {
        this.error = "Не удалось сохранить изменения";
      }
    },
  },
};
</script>

<template>
  <div class="profile-container">
    <div class="profile-card">
      
      <!-- Режим просмотра -->
      <div v-if="!redact">
        <h2 class="profile-title">Мой профиль</h2>
        
        <div class="profile-info">
          <p><strong>Логин:</strong> {{ chlogin }}</p>
          <p><strong>Имя:</strong> {{ chfname || "Не указано" }}</p>
          <p><strong>Фамилия:</strong> {{ chlname || "Не указано" }}</p>
          <p><strong>Роль:</strong> {{ chrole }}</p>
          <p><strong>Дата регистрации:</strong> {{ fmtTime(chcrat) }}</p>
        </div>

        <button type="button" class="btn-main" @click="toredact">
          Редактировать профиль
        </button>
      </div>

      <!-- Режим редактирования -->
      <div v-else>
        <h2 class="profile-title">Редактирование</h2>
        
        <div class="form-group">
          <label>Имя</label>
          <input type="text" v-model="chfname" placeholder="Введите имя" />
        </div>
        
        <div class="form-group">
          <label>Фамилия</label>
          <input type="text" v-model="chlname" placeholder="Введите фамилию" />
        </div>

        <div class="btn-group">
          <button type="button" class="btn-main" @click="change">Сохранить</button>
          <button type="button" class="btn-outline" @click="outredact">Отмена</button>
        </div>
      </div>

      <p v-if="error" class="error-msg">{{ error }}</p>
    </div>
  </div>
</template>

<style scoped>
.profile-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: calc(100vh - 100px);
  padding: 20px;
}

.profile-card {
  background: white;
  padding: 40px;
  border-radius: 24px;
  width: 100%;
  max-width: 450px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.08);
  text-align: center; /* Центрируем текст внутри карточки */
}

.profile-title {
  color: #1D1D1B;
  font-size: 24px;
  margin-top: 0;
  margin-bottom: 24px;
  font-weight: 700;
}

.profile-info {
  margin-bottom: 32px;
  text-align: left; /* Текст данных выравниваем по левому краю */
  background: #F2F3F5;
  padding: 20px;
  border-radius: 16px;
}

.profile-info p {
  margin: 12px 0;
  font-size: 16px;
  color: #333;
  border-bottom: 1px solid #E1E1E1;
  padding-bottom: 12px;
}

.profile-info p:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.profile-info strong {
  color: #1D1D1B;
  font-weight: 600;
}

.form-group {
  text-align: left;
  margin-bottom: 16px;
}

.btn-group {
  display: flex;
  gap: 12px;
  margin-top: 24px;
}

.btn-main {
  flex: 1;
  background: #E30611;
  color: white;
  border: none;
  padding: 16px;
  border-radius: 12px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-main:hover {
  background: #C00000;
}

.btn-outline {
  flex: 1;
  background: white;
  color: #333;
  border: 2px solid #E1E1E1;
  padding: 14px;
  border-radius: 12px;
  font-size: 16px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-outline:hover {
  border-color: #E30611;
  color: #E30611;
}

.error-msg {
  color: #E30611;
  font-weight: 500;
  margin-top: 16px;
}
</style>