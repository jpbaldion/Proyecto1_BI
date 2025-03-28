import { NgModule } from '@angular/core';
import { BrowserModule, provideClientHydration } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { InicioModule } from './Inicio/Inicio.module';
import { PredictModule } from './predict/predict.module';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { PredictComponent } from './predict/predict.component';
import { PredictResultsComponent } from './predict-results/predict-results.component';
import { PredictResultsModule } from './predict-results/predict-results.module';

@NgModule({
  declarations: [
    AppComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    InicioModule,
    PredictModule,
    PredictResultsModule,
    FormsModule,
    HttpClientModule,
    CommonModule
  ],
  providers: [
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
