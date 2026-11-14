<script>
import SignUp from "./components/SignUp.vue";
import SignIn from "./components/SignIn.vue";
import LeaderBoard from "./components/LeaderBoard.vue";
import AppDialogs from "./components/AppDialogs.vue";

export default {
  name: "App",
  components: {
    SignUp,
    SignIn,
    LeaderBoard,
    AppDialogs,
  },
  data() {
    return {
      defurl: "/api",
      activeComp: "SignIn",
      token_type: "",
      access_token: "",
      role: "",
      login: "",
    };
  },
  methods: {
    signOut() {
      this.role = this.type_access_token = "";
      this.activeComp = "SignIn";
    },
    auth(type, at, role, login) {
      this.token_type = type;
      this.access_token = at;
      this.role = role;
      this.login = login;
      // После входа сразу открываем диалоги
      this.activeComp = "AppDialogs"; 
    },
  },
};
</script>

<template>
  <div v-if="role">
    <div id="btnbox">
      <button 
        type="button" 
        :class="{ active: activeComp === 'AppDialogs' }"
        @click="activeComp = 'AppDialogs'">Диалоги</button>
      <button 
        type="button" 
        :class="{ active: activeComp === 'LeaderBoard' }"
        @click="activeComp = 'LeaderBoard'">Мой профиль</button>
      <button 
        type="button" 
        style="margin-left: auto; color: var(--mts-red);"
        @click="signOut">Выйти</button>
    </div>
    
    <component
      :is="activeComp"
      :du="defurl"
      :role="role"
      :token_type="token_type"
      :access_token="access_token"
      :login="login"
    ></component>
  </div>

  <div v-else>
    <!-- Форма входа/регистрации по центру экрана -->
    <div style="display: flex; justify-content: center; align-items: center; height: 100vh; background-color: var(--mts-gray);">
      <div style="width: 100%; max-width: 400px;">
        <div style="text-align: center; margin-bottom: 24px;">
          <!-- Здесь можно добавить логотип или название -->
          <h1 style="color: var(--mts-red); font-weight: 800; font-size: 32px; margin: 0;">Поддержка</h1>
          <p style="color: #666; margin-top: 8px;">Войдите или создайте аккаунт</p>
        </div>
        
        <div id="btnbox" style="justify-content: center; background: transparent; box-shadow: none; border: none; padding: 0 0 20px 0;">
          <button 
            type="button" 
            :class="{ active: activeComp === 'SignIn' }"
            @click="activeComp = 'SignIn'">Вход</button>
          <button 
            type="button" 
            :class="{ active: activeComp === 'SignUp' }"
            @click="activeComp = 'SignUp'">Регистрация</button>
        </div>

        <component
          v-if="activeComp === 'SignIn'"
          :is="activeComp"
          :du="defurl"
          @auth="auth"
        ></component>
        <component v-else :is="activeComp" :du="defurl"></component>
      </div>
    </div>
  </div>
</template>