import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AppService {
  private apiUrl = 'http://127.0.0.1:8000/'
  private predicctionData = []

  constructor(private httpClient: HttpClient) { 

  }

  makePrediction(data: any): Observable<any> {
    return this.httpClient.post<any>(`${this.apiUrl}clasificar/`, data);
  }

  setPredictionData(data: any) {
    this.predicctionData = data;
  }

  getPredictionData() {
    return this.predicctionData;
  }

  reentrenarModelo(data: any[]): Observable<any> {
    return this.httpClient.post(`${this.apiUrl}reentrenar/`, data);
  }
}
