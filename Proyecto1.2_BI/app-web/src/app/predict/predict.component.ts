import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { AppService } from '../app.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-predict',
  templateUrl: './predict.component.html',
  styleUrls: ['./predict.component.css']
})
export class PredictComponent implements OnInit {
  newsTitle: string = '';
  newsDescription: string = '';
  newsList: { titulo: string; descripcion: string, label: number }[] = [];

  constructor(private appService: AppService, private router: Router) {}

  ngOnInit() {}

  addNews() {
    if (this.newsTitle && this.newsDescription) {
      this.newsList.push({
        titulo: this.newsTitle,
        descripcion: this.newsDescription,
        label: 0
      });
      this.newsTitle = '';
      this.newsDescription = '';
    }
  }

  sendNews() {
    this.appService.makePrediction(this.newsList).subscribe(
      (response) => {
        console.log('Predicción recibida:', response.resultados);
        this.appService.setPredictionData(response.resultados);
        this.router.navigate(['/predict-results']);
      },
      (error) => {
        console.error('Error al enviar las noticias:', error);
      }
    );
    console.log('Enviando noticias:', this.newsList);
  }
}
