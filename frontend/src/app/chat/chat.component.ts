import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { NgIf, NgClass } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [NgIf, NgClass, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.scss']
})
export class ChatComponent {
  question: string = '';
  responseData: any = null;
  loading: boolean = false;
  error: string | null = null;

  API_URL = 'http://94.126.205.209/ask';

  constructor(private http: HttpClient) {
    console.log("API_URL в Angular:", this.API_URL);
  }

  askQuestion() {
    if (!this.question.trim()) {
      this.error = 'Введите вопрос';
      return;
    }

    this.loading = true;
    this.error = null;
    this.responseData = null;
    console.log("API_URL перед отправкой запроса:", this.API_URL);
    console.log("Финальный URL запроса:", this.API_URL);
    this.http.post(this.API_URL, { question: this.question }, {
      headers: { 'Content-Type': 'application/json' }
    }).subscribe({
      next: (data) => {
        console.log("Ответ сервера:", data);
        this.responseData = data;
        this.loading = false;
      },
      error: (err) => {
        console.error("Ошибка запроса:", err);
        this.error = 'Ошибка при получении ответа';
        this.loading = false;
      }
    });
  }
}
