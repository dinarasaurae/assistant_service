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

  //API_URL = import.meta.env['VITE_API_URL'] || 'http://localhost:8001';
  API_URL = import.meta.env['VITE_API_URL'] ?? 'http://94.126.205.209:8001';

  constructor(private http: HttpClient) {}

  askQuestion() {
    if (!this.question.trim()) {
      this.error = 'Введите вопрос';
      return;
    }

    this.loading = true;
    this.error = null;
    this.responseData = null;

    this.http.post(`${this.API_URL}/ask`, { question: this.question }).subscribe({
      next: (data) => {
        this.responseData = data;
        this.loading = false;
      },
      error: (err) => {
        this.error = 'Ошибка при получении ответа';
        this.loading = false;
      }
    });
  }
}