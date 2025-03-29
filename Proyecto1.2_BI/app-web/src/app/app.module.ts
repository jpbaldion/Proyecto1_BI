import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { InicioComponent } from './Inicio/Inicio.component';  // Importamos directamente el componente
import { PredictModule } from './predict/predict.module';
import { PredictResultsModule } from './predict-results/predict-results.module';
import { RetrainComponent } from './retrain/retrain.component';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { CommonModule } from '@angular/common';

@NgModule({
  declarations: [
    AppComponent,
    InicioComponent,
    RetrainComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    PredictModule,
    PredictResultsModule,
    FormsModule,
    HttpClientModule,
    CommonModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
